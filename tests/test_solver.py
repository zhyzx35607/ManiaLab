import threading
import warnings

from scipy.optimize import OptimizeWarning

from manialab import arrange, gesture_flow, solver
from manialab.anchor_gestures import _solve_binary as anchor_solve_binary
from manialab.expression import _solve_binary as expression_solve_binary


def test_binary_solver_forces_one_native_thread_and_filters_unsupported_option_warning(monkeypatch):
    result = object()
    captured = {}

    def fake_milp(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        warnings.warn("Unrecognized options detected: threads", RuntimeWarning)
        warnings.warn("solver diagnostic", RuntimeWarning)
        return result

    monkeypatch.setattr(solver, "milp", fake_milp)
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        actual = solver.solve_binary("objective", options={
            "time_limit": 3, "threads": 8, "parallel": True,
        })

    assert actual is result
    assert captured == {
        "args": ("objective",),
        "kwargs": {"options": {"time_limit": 3, "threads": 1, "parallel": False}},
    }
    assert [str(item.message) for item in emitted] == ["solver diagnostic"]


def test_linear_solver_filters_only_unsupported_option_warning():
    result = object()
    captured = {}

    def fake_linprog(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        warnings.warn("Unrecognized options detected: threads", OptimizeWarning)
        warnings.warn("solver diagnostic", OptimizeWarning)
        return result

    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        actual = solver.solve_linear(fake_linprog, "costs", method="highs-ds")

    assert actual is result
    assert captured == {"args": ("costs",), "kwargs": {"method": "highs-ds"}}
    assert [str(item.message) for item in emitted] == ["solver diagnostic"]


def test_binary_and_linear_solver_calls_share_the_native_scheduler_lock(monkeypatch):
    binary_entered = threading.Event()
    release_binary = threading.Event()
    linear_entered = threading.Event()
    active = 0
    maximum_active = 0
    state_lock = threading.Lock()
    errors = []

    def enter_solver():
        nonlocal active, maximum_active
        with state_lock:
            active += 1
            maximum_active = max(maximum_active, active)

    def leave_solver():
        nonlocal active
        with state_lock:
            active -= 1

    def fake_milp(*args, **kwargs):
        enter_solver()
        binary_entered.set()
        try:
            if not release_binary.wait(3):
                raise AssertionError("binary solver was not released")
            return object()
        finally:
            leave_solver()

    def fake_linprog(*args, **kwargs):
        enter_solver()
        linear_entered.set()
        leave_solver()
        return object()

    monkeypatch.setattr(solver, "milp", fake_milp)

    def run_binary():
        try:
            solver.solve_binary("objective")
        except Exception as error:
            errors.append(error)

    def run_linear():
        try:
            solver.solve_linear(fake_linprog, "costs")
        except Exception as error:
            errors.append(error)

    binary_thread = threading.Thread(target=run_binary)
    linear_thread = threading.Thread(target=run_linear)
    linear_started = False
    binary_thread.start()
    binary_ready = binary_entered.wait(2)
    linear_blocked = False
    try:
        if binary_ready:
            linear_thread.start()
            linear_started = True
            linear_blocked = not linear_entered.wait(.05)
    finally:
        release_binary.set()
    binary_thread.join(3)
    if linear_started:
        linear_thread.join(3)

    assert binary_ready
    assert linear_blocked
    assert not binary_thread.is_alive()
    assert not linear_started or not linear_thread.is_alive()
    assert linear_entered.is_set()
    assert maximum_active == 1
    assert errors == []


def test_solver_users_share_the_same_boundary_and_preserve_expression_patch_point():
    assert arrange._solver_lock is solver.solver_lock
    assert gesture_flow.solve_linear is solver.solve_linear
    assert expression_solve_binary is solver.solve_binary
    assert anchor_solve_binary is solver.solve_binary
