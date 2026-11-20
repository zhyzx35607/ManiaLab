"""DNS resolver cancellation remains safe when its request loop has closed."""
import asyncio
import gc
import socket
import threading
import time
from types import SimpleNamespace

import pytest

from manialab import planner as planner_module
from manialab.planner import _BoundedResolver


def test_cancelled_request_prefers_cancel_over_completed_http_error(monkeypatch):
    cancellation = threading.Event()

    async def fail_with_http_error(*args):
        raise planner_module.HTTPStatusError(503)

    monkeypatch.setattr(planner_module, '_post_json', fail_with_http_error)

    async def exercise():
        loop = asyncio.get_running_loop()
        loop_errors = []
        loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))

        async def done_at_completion(tasks, timeout=None):
            await asyncio.sleep(0)
            assert all(task.done() for task in tasks)
            cancellation.set()
            return set(tasks), set()

        monkeypatch.setattr(planner_module.asyncio, 'wait', done_at_completion)
        with pytest.raises(InterruptedError, match='已取消'):
            await planner_module._request_cancellable(
                'https://api.example/v1', b'{}', {}, cancellation.is_set,
            )
        gc.collect()
        await asyncio.sleep(0)
        assert not loop_errors

    asyncio.run(exercise())


@pytest.mark.parametrize('failure_point', ('construction', 'start'))
def test_resolver_recovers_when_worker_thread_setup_fails(monkeypatch, failure_point):
    failures = []
    resolved_hosts = []
    original_thread = threading.Thread
    original_start = original_thread.start

    def getaddrinfo(host, port, family, socktype, proto):
        resolved_hosts.append((host, port))
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, '', ('127.0.0.1', port))]

    resolver = _BoundedResolver(getaddrinfo)

    if failure_point == 'construction':
        def fail_construction_once(*args, **kwargs):
            if kwargs.get('name') == 'manialab-planner-resolver' and not failures:
                failures.append('construction')
                raise RuntimeError('synthetic thread setup failure')
            return original_thread(*args, **kwargs)

        monkeypatch.setattr(planner_module.threading, 'Thread', fail_construction_once)
    else:
        def fail_start_once(thread):
            if thread.name == 'manialab-planner-resolver' and not failures:
                failures.append('start')
                raise RuntimeError('synthetic thread setup failure')
            return original_start(thread)

        monkeypatch.setattr(original_thread, 'start', fail_start_once)

    async def exercise():
        with pytest.raises(RuntimeError, match='synthetic thread setup failure'):
            await resolver.resolve('first.example', 80)

        assert not resolver._busy
        assert resolver._pending is None
        assert resolver._thread is None

        addresses = await resolver.resolve('second.example', 81)
        assert addresses[0][4] == ('127.0.0.1', 81)
        assert resolver._thread is not None

    asyncio.run(exercise())
    assert failures == [failure_point]
    assert resolved_hosts == [('second.example', 81)]


def test_late_dns_result_after_event_loop_close_is_discarded():
    started = threading.Event()
    release = threading.Event()
    callback_attempted = threading.Event()
    callback_errors = []

    def getaddrinfo(host, port, family, socktype, proto):
        started.set()
        assert release.wait(5)
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, '', ('127.0.0.1', port))]

    resolver = _BoundedResolver(getaddrinfo)
    loop = asyncio.new_event_loop()
    original_call_soon_threadsafe = loop.call_soon_threadsafe

    def record_closed_loop(callback, *args, **kwargs):
        try:
            return original_call_soon_threadsafe(callback, *args, **kwargs)
        except RuntimeError as error:
            callback_errors.append(error)
            callback_attempted.set()
            raise

    loop.call_soon_threadsafe = record_closed_loop

    async def cancel_resolution():
        task = asyncio.create_task(resolver.resolve('late.example', 443))
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return task

    try:
        task = loop.run_until_complete(cancel_resolution())
        assert task.cancelled()
        assert resolver._busy
        loop.close()

        release.set()
        assert callback_attempted.wait(2)
        assert len(callback_errors) == 1
        assert isinstance(callback_errors[0], RuntimeError)
        deadline = time.monotonic() + 2
        while resolver._busy and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not resolver._busy
    finally:
        release.set()
        if not loop.is_closed():
            loop.close()


def test_resolved_connection_uses_idna_lookup_and_keeps_tls_hostname(monkeypatch):
    resolved = {}

    class Resolver:
        async def resolve(self, host, port):
            resolved.update(host=host, port=port)
            return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, '', ('192.0.2.10', port))]

    class FakeSocket:
        def __init__(self):
            self.blocking = None

        def setblocking(self, value):
            self.blocking = value

        def close(self):
            pass

    connection = FakeSocket()
    connect = {}
    stream = {}
    context = object()
    monkeypatch.setattr(planner_module, '_RESOLVER', Resolver())
    monkeypatch.setattr(planner_module, 'socket', SimpleNamespace(
        socket=lambda *args: connection,
        AF_INET=socket.AF_INET,
        SOCK_STREAM=socket.SOCK_STREAM,
        IPPROTO_TCP=socket.IPPROTO_TCP,
    ))

    async def exercise():
        loop = asyncio.get_running_loop()

        async def sock_connect(sock, address):
            connect.update(sock=sock, address=address)

        async def open_connection(**kwargs):
            stream.update(kwargs)
            return object(), object()

        monkeypatch.setattr(loop, 'sock_connect', sock_connect)
        monkeypatch.setattr(planner_module.asyncio, 'open_connection', open_connection)
        result = await planner_module._open_resolved_connection(
            'bücher.example', 443, ssl_context=context,
            server_hostname='bücher.example', limit=8192,
        )
        assert len(result) == 2

    asyncio.run(exercise())
    assert resolved == {'host': 'xn--bcher-kva.example', 'port': 443}
    assert connection.blocking is False
    assert connect == {'sock': connection, 'address': ('192.0.2.10', 443)}
    assert stream['sock'] is connection
    assert stream['ssl'] is context
    assert stream['server_hostname'] == 'bücher.example'
