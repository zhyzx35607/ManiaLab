"""Shared SciPy solver scheduling for the application and research tools."""
from __future__ import annotations

import threading
import warnings

from scipy.optimize import OptimizeWarning, milp


# HiGHS owns a process-wide scheduler; app jobs may call it from worker threads.
solver_lock = threading.Lock()


def solve_binary(*args, **kwargs):
    options = dict(kwargs.pop("options", {}), threads=1, parallel=False)
    with solver_lock, warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Unrecognized options detected:.*",
            category=RuntimeWarning,
        )
        return milp(*args, options=options, **kwargs)


def solve_linear(solver, *args, **kwargs):
    with solver_lock, warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Unrecognized options detected:.*",
            category=OptimizeWarning,
        )
        return solver(*args, **kwargs)
