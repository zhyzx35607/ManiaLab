import asyncio
import io
import json
import os
from pathlib import Path
import select
import socket
import socketserver
import ssl
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
import zipfile
import pytest

from manialab.server import create_app
from manialab.domain import Note, export_osu, parse_osu
from manialab import planner as planner_module
from manialab.planner import plan
from manialab.storage import Projects

ROOT=Path(__file__).parents[1]


def fake_mapper_install(data_root):
    root=Path(data_root)/'mapperatorinator'
    python=root/'venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    python.parent.mkdir(parents=True,exist_ok=True);python.touch()
    (root/'ready.json').write_text('{}',encoding='utf-8')


@pytest.mark.parametrize(('method', 'path'), [
    ('get', '/api/jobs/missing'),
    ('post', '/api/jobs/missing/cancel'),
])
def test_unknown_job_returns_not_found(tmp_path, method, path):
    app = create_app(ROOT, tmp_path)
    try:
        response = app.test_client().open(
            path,
            method=method,
            headers={'X-ManiaLab-Token': app.config['TOKEN']},
        )
        assert response.status_code == 404
        assert response.json == {'error': '任务不存在'}
    finally:
        app.extensions['executor'].shutdown()


def test_failed_import_removes_partial_project(tmp_path):
    app=create_app(ROOT,tmp_path);client=app.test_client()
    headers={'X-ManiaLab-Token':app.config['TOKEN']}
    response=client.post('/api/import',data={'files':[
        (io.BytesIO(b'partial-audio'),'song.wav'),
        (io.BytesIO(b'unsupported'),'notes.txt'),
    ]},headers=headers)
    assert response.status_code==400
    assert list((tmp_path/'projects').iterdir())==[]
    app.extensions['executor'].shutdown()


def test_import_removes_project_when_creation_fails_after_parent_mkdir(tmp_path, monkeypatch):
    app = create_app(ROOT, tmp_path)
    client = app.test_client()
    headers = {'X-ManiaLab-Token': app.config['TOKEN']}
    project_root = tmp_path / 'projects'
    original_mkdir = Path.mkdir

    def fail_files_creation(path, *args, **kwargs):
        if path.name == 'files' and path.parent.parent == project_root:
            original_mkdir(path.parent)
            raise OSError('synthetic files directory failure')
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'mkdir', fail_files_creation)
    try:
        response = client.post(
            '/api/import',
            data={'files': (io.BytesIO(b'synthetic audio'), 'song.wav')},
            headers=headers,
        )
        assert response.status_code == 500
        assert list(project_root.iterdir()) == []
    finally:
        app.extensions['executor'].shutdown()


def test_optional_mapper_routes_reject_profiles_without_installer(tmp_path, monkeypatch):
    from manialab.upstream import MapperEngine
    monkeypatch.setattr(MapperEngine, 'status', lambda self: {
        'installed': False, 'model_downloaded': False,
        'install_available': False, 'generate_available': False,
    })
    app=create_app(ROOT,tmp_path);client=app.test_client()
    headers={'X-ManiaLab-Token':app.config['TOKEN']}
    try:
        state=client.post('/api/import',data={
            'files':(io.BytesIO(b'not decoded until analysis'),'song.wav')
        },headers=headers).json
        base=f"/api/projects/{state['id']}"
        install=client.post(base+'/engine-install',json={},headers=headers)
        generate=client.post(base+'/engine-generate',json={},headers=headers)
        assert install.status_code==400 and '不含原模型安装器' in install.json['error']
        assert generate.status_code==400 and '不含原模型运行器' in generate.json['error']
    finally:
        app.extensions['executor'].shutdown()


def test_full_remap_export(tmp_path):
    app=create_app(ROOT,tmp_path); client=app.test_client(); headers={'X-ManiaLab-Token':app.config['TOKEN']}
    original=export_osu([Note(1000,4000,0),Note(1000,5000,3),Note(2000,2000,8)],10,ROOT/'vendor/mapperatorinator/template.osu',audio='song.wav')
    data={'files':[(io.BytesIO(original.encode()),'chart.osu'),(io.BytesIO(b'fake-audio'),'song.wav')]}
    response=client.post('/api/import',data=data,headers=headers)
    assert response.status_code==200,response.json
    pid=response.json['id']
    denied=client.post(f'/api/projects/{pid}/generate',json={})
    assert denied.status_code==403
    response=client.post(f'/api/projects/{pid}/generate',json={'mode':'remap','settings':{'keys':10}},headers=headers)
    jid=response.json['id']
    for _ in range(100):
        job=client.get('/api/jobs/'+jid).json
        if job['status'] in ('done','error'):break
        time.sleep(.02)
    assert job['status']=='done',job
    exported=client.get(f'/api/projects/{pid}/export/osu')
    assert exported.status_code==200
    parsed=parse_osu(exported.data.decode())
    assert [(n.start,n.end) for n in parsed['notes']]==[(1000,4000),(1000,5000),(2000,2000)]
    package=client.get(f'/api/projects/{pid}/export/osz')
    with zipfile.ZipFile(io.BytesIO(package.data)) as z:
        assert set(z.namelist())=={'ManiaLab.osu','song.wav'}
    assert client.get('/api/status',headers={'Host':'evil.example'}).status_code==403
    assert client.get('/api/status',headers={'Origin':'https://evil.example'}).status_code==403
    app.extensions['executor'].shutdown()


def test_real_http_planner_mock():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            assert self.path=='/v1/chat/completions'
            assert 'segments' in body['messages'][1]['content']
            result={'choices':[{'message':{'content':json.dumps({'preferences':{'ln_ratio':.8},'explanation':'test'})}}]}
            self.send_response(200);self.end_headers();self.wfile.write(json.dumps(result).encode())
        def log_message(self,*args):pass
    server=HTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        result=plan(f'http://127.0.0.1:{server.server_port}/v1','test','fake-test-key','prefer sustain',[{'start_ms':0,'end_ms':1000}],{})
        assert result['preferences']['ln_ratio']==.8
    finally:server.shutdown();thread.join();server.server_close()


@pytest.mark.parametrize(
    ('status', 'location', 'message'),
    [
        (401, None, 'HTTP 401'),
        (302, 'https://127.0.0.1/redirected', '重定向'),
    ],
)
def test_real_http_planner_rejects_status_and_redirect(status, location, message):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(status)
            if location:
                self.send_header('Location', location)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(ValueError, match=message):
            plan(
                f'http://127.0.0.1:{server.server_port}/v1',
                'test',
                'fake-test-key',
                'prefer sustain',
                [{'start_ms': 0, 'end_ms': 1000}],
                {},
            )
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_real_http_planner_accepts_chunked_response():
    result = {
        'choices': [{'message': {'content': json.dumps({
            'preferences': {'density': 12},
            'segments': [],
            'explanation': 'chunked fixture',
        })}}],
    }
    payload = json.dumps(result).encode()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            chunk = f'{len(payload):X}\r\n'.encode() + payload + b'\r\n0\r\n\r\n'
            self.send_response(200)
            self.send_header('Transfer-Encoding', 'chunked')
            self.send_header('Connection', 'close')
            self.end_headers()
            self.wfile.write(chunk)

        def log_message(self, *args):
            pass

    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        response = plan(
            f'http://127.0.0.1:{server.server_port}/v1',
            'test',
            '',
            'prefer sustain',
            [{'start_ms': 0, 'end_ms': 1000}],
            {},
        )
        assert response['preferences']['density'] == 12
        assert response['explanation'] == 'chunked fixture'
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_real_http_planner_skips_informational_response():
    result = {
        'choices': [{'message': {'content': json.dumps({
            'preferences': {'density': 13},
            'segments': [],
            'explanation': 'after continue',
        })}}],
    }
    payload = json.dumps(result).encode()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            self.wfile.write(b'HTTP/1.1 100 Continue\r\n\r\n')
            self.wfile.flush()
            self.send_response(200)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Connection', 'close')
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        response = plan(
            f'http://127.0.0.1:{server.server_port}/v1',
            'test',
            '',
            'prefer sustain',
            [{'start_ms': 0, 'end_ms': 1000}],
            {},
        )
        assert response['preferences']['density'] == 13
        assert response['explanation'] == 'after continue'
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_real_http_planner_uses_authenticated_http_proxy(monkeypatch):
    result = {
        'choices': [{'message': {'content': json.dumps({
            'preferences': {'density': 14},
            'segments': [],
            'explanation': 'through proxy',
        })}}],
    }
    payload = json.dumps(result).encode()
    observed = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            observed['path'] = self.path
            observed['authorization'] = self.headers.get('Proxy-Authorization')
            self.send_response(200)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Connection', 'close')
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(planner_module.urllib.request, 'proxy_bypass', lambda host: False)
    monkeypatch.setattr(
        planner_module.urllib.request,
        'getproxies',
        lambda: {'http': f'http://proxy%40user:secret%3Apass@127.0.0.1:{server.server_port}'},
    )
    try:
        response = plan(
            'http://127.0.0.1:12345/v1',
            'test',
            '',
            'prefer sustain',
            [{'start_ms': 0, 'end_ms': 1000}],
            {},
        )
        assert response['preferences']['density'] == 14
        assert observed['path'] == 'http://127.0.0.1:12345/v1/chat/completions'
        assert observed['authorization'] == 'Basic cHJveHlAdXNlcjpzZWNyZXQ6cGFzcw=='
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_real_http_planner_tunnels_through_https_proxy(monkeypatch):
    result = {
        'choices': [{'message': {'content': json.dumps({
            'preferences': {'density': 15},
            'segments': [],
            'explanation': 'through secure proxy',
        })}}],
    }
    payload = json.dumps(result).encode()
    response_bytes = (
        b'HTTP/1.1 200 Connection Established\r\n\r\n'
        + b'HTTP/1.1 200 OK\r\n'
        + f'Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n'.encode()
        + payload
    )

    class Writer:
        def __init__(self):
            self.writes = []
            self.start_tls_server_name = None
            self.closed = False

        def write(self, value):
            self.writes.append(value)

        async def drain(self):
            pass

        async def start_tls(self, context, server_hostname=None):
            self.start_tls_server_name = server_hostname

        def close(self):
            self.closed = True

        async def wait_closed(self):
            pass

    writer = Writer()
    connection = {}

    async def open_resolved_connection(host, port, *, ssl_context, server_hostname, limit):
        reader = asyncio.StreamReader()
        reader.feed_data(response_bytes)
        reader.feed_eof()
        connection.update(
            host=host, port=port, ssl_context=ssl_context,
            server_hostname=server_hostname, limit=limit,
        )
        return reader, writer

    monkeypatch.setattr(planner_module.urllib.request, 'proxy_bypass', lambda host: False)
    monkeypatch.setattr(
        planner_module.urllib.request,
        'getproxies',
        lambda: {'https': 'https://proxy%40user:secret%3Apass@proxy.example:8443'},
    )
    monkeypatch.setattr(planner_module, '_open_resolved_connection', open_resolved_connection)
    response = plan(
        'https://api.example/v1',
        'test',
        'api-key',
        'prefer sustain',
        [{'start_ms': 0, 'end_ms': 1000}],
        {},
    )
    assert response['preferences']['density'] == 15
    assert connection['host'] == 'proxy.example'
    assert connection['port'] == 8443
    assert connection['ssl_context'] is not None
    assert connection['ssl_context'].check_hostname
    assert connection['ssl_context'].verify_mode == planner_module.ssl.CERT_REQUIRED
    assert connection['server_hostname'] == 'proxy.example'
    assert writer.start_tls_server_name == 'api.example'
    assert writer.closed
    connect_request, tunneled_request = writer.writes
    assert connect_request.startswith(b'CONNECT api.example:443 HTTP/1.1\r\n')
    assert b'Proxy-Authorization: Basic cHJveHlAdXNlcjpzZWNyZXQ6cGFzcw==\r\n' in connect_request
    assert tunneled_request.startswith(b'POST /v1/chat/completions HTTP/1.1\r\n')
    assert b'Authorization: Bearer api-key\r\n' in tunneled_request
    assert b'Host: api.example\r\n' in tunneled_request
    assert b'Proxy-Authorization:' not in tunneled_request


def test_http_planner_uses_real_tls_over_tls_https_proxy(monkeypatch):
    fixture_dir = Path(__file__).parent / 'fixtures' / 'https_proxy_tls'
    ca_file = fixture_dir / 'ca.pem'
    response = {
        'choices': [{'message': {'content': json.dumps({
            'preferences': {'density': 16},
            'segments': [],
            'explanation': 'through local TLS tunnel',
        })}}],
    }
    response_bytes = json.dumps(response).encode()
    observed = {}
    proxy_failures = []

    class OriginHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            observed['origin_path'] = self.path
            observed['origin_host'] = self.headers.get('Host')
            observed['origin_authorization'] = self.headers.get('Authorization')
            observed['origin_proxy_authorization'] = self.headers.get('Proxy-Authorization')
            self.send_response(200)
            self.send_header('Content-Length', str(len(response_bytes)))
            self.send_header('Connection', 'close')
            self.end_headers()
            self.wfile.write(response_bytes)

        def log_message(self, *args):
            pass

    origin = ThreadingHTTPServer(('127.0.0.1', 0), OriginHandler)
    origin_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    origin_context.load_cert_chain(
        fixture_dir / 'origin-cert.pem', fixture_dir / 'origin-key.pem'
    )
    origin.socket = origin_context.wrap_socket(origin.socket, server_side=True)
    origin_thread = threading.Thread(target=origin.serve_forever, daemon=True)
    origin_thread.start()

    class TunnelHandler(socketserver.StreamRequestHandler):
        def handle(self):
            try:
                request_line = self.rfile.readline().decode('ascii').rstrip('\r\n')
                headers = {}
                while True:
                    line = self.rfile.readline()
                    if line in (b'\r\n', b'\n', b''):
                        break
                    name, value = line.decode('latin-1').split(':', 1)
                    headers[name.lower()] = value.strip()
                observed['connect_line'] = request_line
                observed['connect_host'] = headers.get('host')
                observed['connect_proxy_authorization'] = headers.get('proxy-authorization')

                with socket.create_connection(('127.0.0.1', origin.server_port), timeout=5) as upstream:
                    self.request.settimeout(5)
                    upstream.settimeout(5)
                    self.wfile.write(b'HTTP/1.1 200 Connection Established\r\n\r\n')
                    self.wfile.flush()
                    while True:
                        readable, _, _ = select.select((self.request, upstream), (), (), 5)
                        if not readable:
                            raise TimeoutError('local HTTPS proxy tunnel stalled')
                        for source in readable:
                            data = source.recv(65536)
                            if not data:
                                return
                            destination = upstream if source is self.request else self.request
                            destination.sendall(data)
            except (OSError, ssl.SSLError, TimeoutError, UnicodeError, ValueError) as error:
                proxy_failures.append(error)

    class TLSProxyServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

        def __init__(self, address, handler, context):
            self.tls_context = context
            super().__init__(address, handler)

        def get_request(self):
            connection, address = super().get_request()
            try:
                return self.tls_context.wrap_socket(connection, server_side=True), address
            except BaseException:
                connection.close()
                raise

    proxy_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    proxy_context.load_cert_chain(
        fixture_dir / 'proxy-cert.pem', fixture_dir / 'proxy-key.pem'
    )
    proxy = TLSProxyServer(('127.0.0.1', 0), TunnelHandler, proxy_context)
    proxy_port = proxy.server_address[1]
    proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()

    monkeypatch.setenv('SSL_CERT_FILE', str(ca_file))
    monkeypatch.setattr(planner_module.urllib.request, 'proxy_bypass', lambda host: False)
    monkeypatch.setattr(
        planner_module.urllib.request,
        'getproxies',
        lambda: {
            'https': f'https://proxy%40user:secret%3Apass@127.0.0.1:{proxy_port}'
        },
    )
    try:
        result = plan(
            f'https://api.test:{origin.server_port}/v1',
            'test',
            'api-key',
            'prefer sustain',
            [{'start_ms': 0, 'end_ms': 1000}],
            {},
        )
        assert result['preferences']['density'] == 16
        assert observed['connect_line'] == f'CONNECT api.test:{origin.server_port} HTTP/1.1'
        assert observed['connect_host'] == f'api.test:{origin.server_port}'
        assert observed['connect_proxy_authorization'] == 'Basic cHJveHlAdXNlcjpzZWNyZXQ6cGFzcw=='
        assert observed['origin_path'] == '/v1/chat/completions'
        assert observed['origin_host'] == f'api.test:{origin.server_port}'
        assert observed['origin_authorization'] == 'Bearer api-key'
        assert observed['origin_proxy_authorization'] is None
        assert proxy_failures == []
    finally:
        proxy.shutdown()
        proxy.server_close()
        proxy_thread.join(timeout=2)
        origin.shutdown()
        origin.server_close()
        origin_thread.join(timeout=2)


def test_bounded_resolver_cancellation_does_not_use_default_executor(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    calls = []

    def getaddrinfo(host, port, family, socktype, proto):
        calls.append((host, port, threading.current_thread()))
        if len(calls) == 1:
            started.set()
            assert release.wait(5)
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, '', ('127.0.0.1', port))]

    resolver = planner_module._BoundedResolver(getaddrinfo)

    async def exercise():
        loop = asyncio.get_running_loop()
        loop_errors = []
        loop.set_exception_handler(lambda loop, context: loop_errors.append(context))

        def reject_executor(*args, **kwargs):
            raise AssertionError('DNS resolution must not use asyncio default executor')

        monkeypatch.setattr(loop, 'run_in_executor', reject_executor)
        first = asyncio.create_task(resolver.resolve('xn--bcher-kva.example', 80))
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

        with pytest.raises(OSError, match='busy'):
            await resolver.resolve('second.example', 80)
        assert len(calls) == 1
        assert resolver._busy
        resolver_thread = resolver._thread
        assert resolver_thread.name == 'manialab-planner-resolver'
        assert calls[0][2] is resolver_thread

        release.set()
        for _ in range(100):
            if not resolver._busy:
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0)
        assert not loop_errors
        addresses = await resolver.resolve('third.example', 81)
        assert addresses[0][4] == ('127.0.0.1', 81)
        assert resolver._thread is resolver_thread
        assert len(calls) == 2
        assert first.cancelled()

    try:
        asyncio.run(exercise())
    finally:
        release.set()


def test_cancelled_plan_during_blocked_dns_releases_worker_and_preserves_project(tmp_path, monkeypatch):
    resolver_started = threading.Event()
    release_resolver = threading.Event()
    resolution_calls = []
    original_getaddrinfo = socket.getaddrinfo

    def blocked_getaddrinfo(host, port, family, socktype, proto):
        resolution_calls.append((host, port))
        if len(resolution_calls) == 1:
            resolver_started.set()
            assert release_resolver.wait(5)
        return original_getaddrinfo(host, port, family, socktype, proto)

    resolver = planner_module._BoundedResolver(blocked_getaddrinfo)
    monkeypatch.setattr(planner_module, '_RESOLVER', resolver)
    monkeypatch.setattr(planner_module.urllib.request, 'proxy_bypass', lambda host: True)
    response_payload = json.dumps({
        'choices': [{'message': {'content': json.dumps({
            'preferences': {'density': 12},
            'segments': [],
            'explanation': 'after DNS release',
        })}}],
    }).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200)
            self.send_header('Content-Length', str(len(response_payload)))
            self.send_header('Connection', 'close')
            self.end_headers()
            self.wfile.write(response_payload)

        def log_message(self, *args):
            pass

    http_server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    http_thread = threading.Thread(target=http_server.serve_forever, daemon=True)
    http_thread.start()
    app = create_app(ROOT, tmp_path)
    client = app.test_client()
    headers = {'X-ManiaLab-Token': app.config['TOKEN']}
    endpoint = f'http://localhost:{http_server.server_port}/v1'
    request = {
        'endpoint': endpoint,
        'model': 'blocked-dns',
        'instruction': 'Use deterministic fixture preferences',
        'settings': {},
    }
    resolver_threads_before = sum(
        thread.name == 'manialab-planner-resolver' for thread in threading.enumerate()
    )

    def wait_for_job(job_id, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = client.get(f'/api/jobs/{job_id}').json
            if job['status'] in ('done', 'error', 'cancelled'):
                return job
            time.sleep(0.01)
        pytest.fail(f'job {job_id} did not finish: {client.get(f"/api/jobs/{job_id}").json}')

    try:
        imported = client.post(
            '/api/import',
            data={'files': (io.BytesIO(b'synthetic audio'), 'song.wav')},
            headers=headers,
        )
        assert imported.status_code == 200
        project_id = imported.json['id']
        projects = app.extensions['projects']
        state = projects.load(project_id)
        state['analysis'] = {'segments': [{'start_ms': 0, 'end_ms': 1000}]}
        state['plan'] = {'sentinel': 'previous'}
        projects.save(state)
        previous = projects.load(project_id)

        submitted = client.post(f'/api/projects/{project_id}/plan', json=request, headers=headers)
        assert submitted.status_code == 200
        first_job_id = submitted.json['id']
        assert resolver_started.wait(5)
        resolver_thread = resolver._thread

        started_at = time.monotonic()
        cancelled = client.post(f'/api/jobs/{first_job_id}/cancel', headers=headers)
        assert cancelled.status_code == 200
        assert wait_for_job(first_job_id)['status'] == 'cancelled'
        assert time.monotonic() - started_at < 2
        assert projects.load(project_id) == previous
        assert resolver._thread is resolver_thread

        busy_submission = client.post(f'/api/projects/{project_id}/plan', json=request, headers=headers)
        assert busy_submission.status_code == 200
        busy_job = wait_for_job(busy_submission.json['id'])
        assert busy_job['status'] == 'error'
        assert len(resolution_calls) == 1
        assert resolver._thread is resolver_thread

        release_resolver.set()
        deadline = time.monotonic() + 2
        while resolver._busy and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not resolver._busy

        resumed = client.post(f'/api/projects/{project_id}/plan', json=request, headers=headers)
        assert resumed.status_code == 200
        resumed_job = wait_for_job(resumed.json['id'])
        assert resumed_job['status'] == 'done', resumed_job
        assert projects.load(project_id)['plan']['explanation'] == 'after DNS release'
        assert resolver._thread is resolver_thread
        assert len(resolution_calls) == 2
        assert sum(
            thread.name == 'manialab-planner-resolver' for thread in threading.enumerate()
        ) == resolver_threads_before + 1
    finally:
        release_resolver.set()
        app.extensions['executor'].shutdown()
        http_server.shutdown()
        http_thread.join()
        http_server.server_close()


def test_planner_rejects_control_characters_in_endpoint():
    with pytest.raises(ValueError, match='API 地址格式无效'):
        plan(
            'http://127.0.0.1/v1\r\nX-Injected: yes',
            'test',
            '',
            'prefer sustain',
            [{'start_ms': 0, 'end_ms': 1000}],
            {},
        )


@pytest.mark.parametrize('endpoint', [
    'https://api.example:0/v1',
    'https://api.example:65536/v1',
])
def test_planner_rejects_invalid_endpoint_port(endpoint):
    with pytest.raises(ValueError, match='API 地址格式无效'):
        plan(
            endpoint,
            'test',
            '',
            'prefer sustain',
            [{'start_ms': 0, 'end_ms': 1000}],
            {},
        )


@pytest.mark.parametrize('proxy', [
    'http://127.0.0.1:0',
    'http://127.0.0.1:65536',
])
def test_planner_rejects_invalid_proxy_port(monkeypatch, proxy):
    monkeypatch.setattr(planner_module.urllib.request, 'proxy_bypass', lambda host: False)
    monkeypatch.setattr(planner_module.urllib.request, 'getproxies', lambda: {'http': proxy})
    with pytest.raises(ValueError, match='API 代理地址格式无效'):
        plan(
            'http://127.0.0.1/v1',
            'test',
            '',
            'prefer sustain',
            [{'start_ms': 0, 'end_ms': 1000}],
            {},
        )


def test_cancelled_http_planner_releases_task_worker(tmp_path):
    first_request_received = threading.Event()
    release_first_request = threading.Event()
    request_lock = threading.Lock()
    request_count = 0

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            nonlocal request_count
            self.rfile.read(int(self.headers['Content-Length']))
            with request_lock:
                request_count += 1
                current_request = request_count
            if current_request == 1:
                first_request_received.set()
                release_first_request.wait(5)
            payload = json.dumps({
                'choices': [{'message': {'content': json.dumps({
                    'preferences': {'density': 11},
                    'segments': [],
                    'explanation': 'cancellable fixture',
                })}}],
            }).encode()
            try:
                self.send_response(200)
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except OSError:
                pass

        def log_message(self, *args):
            pass

    http_server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    http_thread = threading.Thread(target=http_server.serve_forever, daemon=True)
    http_thread.start()
    app = create_app(ROOT, tmp_path)
    client = app.test_client()
    headers = {'X-ManiaLab-Token': app.config['TOKEN']}
    try:
        imported = client.post(
            '/api/import',
            data={'files': (io.BytesIO(b'synthetic audio'), 'song.wav')},
            headers=headers,
        )
        assert imported.status_code == 200
        project_id = imported.json['id']
        projects = app.extensions['projects']
        state = projects.load(project_id)
        state['analysis'] = {'segments': [{'start_ms': 0, 'end_ms': 1000}]}
        state['plan'] = {'sentinel': 'previous'}
        projects.save(state)
        previous = projects.load(project_id)
        endpoint = f'http://127.0.0.1:{http_server.server_port}/v1'
        request = {
            'endpoint': endpoint,
            'model': 'cancel',
            'instruction': 'Use deterministic fixture preferences',
            'settings': {},
        }

        submitted = client.post(f'/api/projects/{project_id}/plan', json=request, headers=headers)
        assert submitted.status_code == 200
        job_id = submitted.json['id']
        assert first_request_received.wait(5)

        started = time.monotonic()
        cancelled = client.post(f'/api/jobs/{job_id}/cancel', headers=headers)
        assert cancelled.status_code == 200
        deadline = started + 2
        while time.monotonic() < deadline:
            job = client.get(f'/api/jobs/{job_id}').json
            if job['status'] in ('done', 'error', 'cancelled'):
                break
            time.sleep(0.01)
        assert job['status'] == 'cancelled'
        assert time.monotonic() - started < 2
        assert projects.load(project_id) == previous

        resumed = client.post(
            f'/api/projects/{project_id}/plan',
            json=request | {'model': 'success'},
            headers=headers,
        )
        assert resumed.status_code == 200
        resumed_id = resumed.json['id']
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            resumed_job = client.get(f'/api/jobs/{resumed_id}').json
            if resumed_job['status'] in ('done', 'error', 'cancelled'):
                break
            time.sleep(0.01)
        assert resumed_job['status'] == 'done', resumed_job
        assert projects.load(project_id)['plan']['explanation'] == 'cancellable fixture'
    finally:
        release_first_request.set()
        app.extensions['executor'].shutdown()
        http_server.shutdown()
        http_thread.join()
        http_server.server_close()


def test_audio_generation_uses_confirmed_grid_and_real_ln_target(tmp_path):
    app = create_app(ROOT, tmp_path)
    client = app.test_client()
    headers = {'X-ManiaLab-Token': app.config['TOKEN']}
    state = client.post('/api/import', data={'files': (io.BytesIO(b'placeholder'), 'song.wav')}, headers=headers).json
    pid = state['id']
    notes = [Note(1014 + i * 1000, 1764 + i * 1000, source='pitch', id=str(i)) for i in range(20)]
    state['analysis'] = {'revision': 3, 'candidates': [n.data() for n in notes], 'tempo': {'bpm': 121, 'offset': 20, 'stable': False}}
    Projects(tmp_path / 'projects').save(state)
    payload = {'mode': 'audio', 'settings': {'ln_ratio': .5}, 'bpm': 120, 'offset': 0, 'division': 4,
               'snap': True, 'timing_confirmed': True}
    job = client.post(f'/api/projects/{pid}/generate', json=payload, headers=headers).json
    for _ in range(300):
        result = client.get('/api/jobs/' + job['id']).json
        if result['status'] in ('done', 'error'):
            break
        time.sleep(.02)
    assert result['status'] == 'done', result
    output = client.get(f'/api/projects/{pid}').json['variants'][0]
    assert output['report']['ln'] == 10
    assert output['selection']['alignment']['moved'] == 20
    assert all(n['start'] % 1000 == 0 for n in output['notes'])
    parsed = parse_osu(client.get(f'/api/projects/{pid}/export/osu').data.decode())
    assert '0.000,500.00000000,4,2,0,70,1,0' in parsed['timing']
    assert output['report']['grid']['off_grid'] == 0
    assert client.post(f'/api/projects/{pid}/timing-check', json={'bpm': 0}, headers=headers).status_code == 400
    app.extensions['executor'].shutdown()


def wait_job(client, job):
    for _ in range(500):
        result=client.get('/api/jobs/'+job['id']).json
        if result['status'] in ('done','error','cancelled'):
            return result
        time.sleep(.02)
    raise AssertionError('job timed out')


def test_http_cancel_waits_for_project_commit_and_preserves_done_result(tmp_path, monkeypatch):
    app = create_app(ROOT, tmp_path)
    client = app.test_client()
    headers = {'X-ManiaLab-Token': app.config['TOKEN']}
    imported = client.post(
        '/api/import',
        data={'files': (io.BytesIO(b'synthetic audio'), 'song.wav')},
        headers=headers,
    )
    assert imported.status_code == 200
    project_id = imported.json['id']
    projects = app.extensions['projects']
    service = app.extensions['task_service']
    save_entered, release_save = threading.Event(), threading.Event()
    cancel_lock_contended, cancel_finished = threading.Event(), threading.Event()
    original_save = projects.save
    original_cancel = service.cancel
    cancel_result = {}
    cancel_thread = None
    cancel_thread_id = [None]

    class ObservedRLock:
        def __init__(self, lock):
            self.lock = lock

        def acquire(self, blocking=True, timeout=-1):
            if (blocking and timeout == -1 and
                    threading.get_ident() == cancel_thread_id[0]):
                if not self.lock.acquire(blocking=False):
                    cancel_lock_contended.set()
                    return self.lock.acquire()
                return True
            if timeout == -1:
                return self.lock.acquire(blocking)
            return self.lock.acquire(blocking, timeout)

        def release(self):
            return self.lock.release()

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, *_):
            self.release()

    service.lock = ObservedRLock(service.lock)

    def gated_save(state):
        if state.get('transaction_marker') == 'committed':
            save_entered.set()
            assert release_save.wait(5)
        return original_save(state)

    def observed_cancel(job_id):
        cancel_thread_id[0] = threading.get_ident()
        return original_cancel(job_id)

    monkeypatch.setattr(projects, 'save', gated_save)
    monkeypatch.setattr(service, 'cancel', observed_cancel)

    def operation(state, progress, cancelled):
        state['transaction_marker'] = 'committed'
        return {'saved': True}

    try:
        job = service.submit(project_id, operation)
        assert save_entered.wait(5)

        def request_cancel():
            try:
                with app.test_client() as cancel_client:
                    response = cancel_client.post(
                        f"/api/jobs/{job['id']}/cancel", headers=headers
                    )
                    cancel_result.update(status=response.status_code, body=response.json)
            except BaseException as error:
                cancel_result['error'] = error
            finally:
                cancel_finished.set()

        cancel_thread = threading.Thread(target=request_cancel, daemon=True)
        cancel_thread.start()
        assert cancel_lock_contended.wait(5)

        release_save.set()
        cancel_thread.join(timeout=5)
        assert not cancel_thread.is_alive()
        assert cancel_result == {'status': 200, 'body': {'ok': True}}

        result = wait_job(client, job)
        assert result['status'] == 'done'
        assert result['result'] == {'saved': True}
        assert result['cancel'] is False
        assert projects.load(project_id)['transaction_marker'] == 'committed'
        assert service._idle.is_set()
    finally:
        release_save.set()
        if cancel_thread is not None:
            cancel_thread.join(timeout=5)
        app.extensions['executor'].shutdown()


def test_generation_applies_local_recommendation_expression_to_export(tmp_path):
    from manialab.arrange import parameters
    from manialab.recommendations import evidence_signature

    app=create_app(ROOT,tmp_path)
    client=app.test_client()
    headers={'X-ManiaLab-Token':app.config['TOKEN']}
    state=client.post('/api/import',data={'files':(io.BytesIO(b'placeholder'),'song.wav')},headers=headers).json
    pid=state['id'];base=f'/api/projects/{pid}'
    prefs=parameters({'keys':10,'ln_ratio':.55,'min_ln':240})
    notes=[Note(1000,1600,pitch=60,source='pitch',voice='vocals',id='early').data(),
           Note(7000,7600,pitch=64,source='pitch',voice='vocals',id='late').data()]
    analysis=dict(revision=3,duration=12.,tempo=dict(bpm=120,offset=0,division=4,stable=True),
        candidates=notes,phrase_candidates=[],timing_attacks=[],
        structure=dict(segments=[dict(start_ms=0,end_ms=6000),dict(start_ms=6000,end_ms=12000)]))
    segments=[
        dict(start_ms=0,end_ms=6000,preferences=dict(ln_ratio=0,min_ln=100)),
        dict(start_ms=6000,end_ms=12000,preferences=dict(ln_ratio=.9,min_ln=500)),
    ]
    state['settings']=prefs
    state['analysis']=analysis
    state['plan']=dict(source='local-audio-recommendation',preferences=prefs,
        segments=segments,keys=10,bpm=120,evidence_signature=evidence_signature(analysis))
    Projects(tmp_path/'projects').save(state)

    try:
        job=client.post(base+'/generate',json=dict(mode='audio',settings=prefs,bpm=120,
            offset=0,division=4,timing_confirmed=True),headers=headers).json
        result=wait_job(client,job)
        assert result['status']=='done',result

        variant=client.get(base).json['variants'][-1]
        placed={n['id']:(n['start'],n['end']-n['start']) for n in variant['notes']}
        assert placed=={'early':(1000,0),'late':(7000,625)}
        exported=parse_osu(client.get(base+'/export/osu').data.decode())
        assert sorted((n.start,n.end) for n in exported['notes'])==[(1000,1000),(7000,7625)]
    finally:
        app.extensions['executor'].shutdown()


def test_original_model_requires_confirmation_and_preserves_provided_clock(tmp_path,monkeypatch):
    from manialab.upstream import MapperEngine
    app=create_app(ROOT,tmp_path);client=app.test_client();headers={'X-ManiaLab-Token':app.config['TOKEN']}
    fake_mapper_install(tmp_path)
    projects=Projects(tmp_path/'projects')
    state=client.post('/api/import',data={'files':(io.BytesIO(b'placeholder'),'song.wav')},headers=headers).json
    pid=state['id'];base=f'/api/projects/{pid}'
    (projects.path(pid)/'preview.wav').write_bytes(b'placeholder')
    seen=[]
    def model(self,audio,output,prefs,progress,cancelled,seconds,tempo):
        seen.append(tempo)
        output.mkdir(parents=True);target=output/'generated.osu'
        target.write_text(export_osu([Note(1092,2092,0,id='a'),Note(2595,2595,1,id='b')],10,
            ROOT/'vendor/mapperatorinator/template.osu',bpm=tempo['bpm'],offset=tempo['offset']),encoding='utf-8')
        return target
    monkeypatch.setattr(MapperEngine,'generate',model)
    monkeypatch.setattr('manialab.generation.assign_lanes',lambda *a,**k:pytest.fail('Model layout was overwritten'))
    payload=dict(use_timing=True,bpm=120,offset=93,division=4,audio_shift_ms=22.5,settings={'keys':10})
    try:
        rejected=wait_job(client,client.post(base+'/engine-generate',json=payload,headers=headers).json)
        assert rejected['status']=='error' and '确认' in rejected['message'] and not seen,rejected
        result=wait_job(client,client.post(base+'/engine-generate',json=payload|{'timing_confirmed':True},headers=headers).json)
        assert result['status']=='done',result
        saved=client.get(base).json;v=saved['variants'][0]
        assert seen[0]['bpm']==120 and seen[0]['offset']==93
        assert v['report']['ln_patterns'] and v['provided_timing']['offset']==93
        assert v['report']['grid']['off_grid']==0
        assert v['selection']['alignment']['moved']==2 and v['division']==4
        assert v['notes'][0]['evidence_start']==1092
        assert [n['lane'] for n in v['notes']]==[0,1]
        assert v['selection']['layout_origin']=='mapperatorinator'
        exported=parse_osu(client.get(base+'/export/osu').data.decode())
        assert [(n.start,n.end) for n in exported['notes']]==[(1093,2093),(2593,2593)]
        assert '93.000,500.00000000,4,2,0,70,1,0' in exported['timing']
        saved=projects.load(pid)
        saved['variants'][0]['notes'][0]['raw']=saved['variants'][0]['notes'][0]['raw'].replace(',1093,128,',',1092,128,')
        projects.save(saved)
        response=client.get(base+'/export/osu')
        assert response.status_code==400 and '落线' in response.json['error']
        original_audio=client.post(base+'/chart',json={'path':''},headers=headers).json
        assert original_audio['original'] is None and original_audio['mode']=='audio'
        assert original_audio['variants']==state['variants']
        model_chart=client.post(base+'/chart',json={'path':saved['original']},headers=headers).json
        assert model_chart['variants']==saved['variants'] and model_chart['active']==0
    finally:app.extensions['executor'].shutdown()


def test_original_model_conflicts_are_reported_without_silent_reassignment(tmp_path,monkeypatch):
    from manialab.upstream import MapperEngine
    app=create_app(ROOT,tmp_path);client=app.test_client();headers={'X-ManiaLab-Token':app.config['TOKEN']}
    fake_mapper_install(tmp_path)
    projects=Projects(tmp_path/'projects')
    state=client.post('/api/import',data={'files':(io.BytesIO(b'placeholder'),'song.wav')},headers=headers).json
    pid=state['id'];base=f'/api/projects/{pid}'
    (projects.path(pid)/'preview.wav').write_bytes(b'placeholder')
    def model(self,audio,output,prefs,progress,cancelled,seconds,tempo):
        text=export_osu([Note(1000,4000,0),Note(2000,2000,1)],10,ROOT/'vendor/mapperatorinator/template.osu')
        text=text.replace('76,192,2000,1,','25,192,2000,1,')
        output.mkdir();target=output/'generated.osu';target.write_text(text,encoding='utf-8')
        return target
    monkeypatch.setattr(MapperEngine,'generate',model)
    monkeypatch.setattr('manialab.generation.assign_lanes',lambda *a,**k:pytest.fail('Model conflict was hidden'))
    try:
        result=wait_job(client,client.post(base+'/engine-generate',json={'settings':{'keys':10}},headers=headers).json)
        assert result['status']=='done',result
        variant=client.get(base).json['variants'][0]
        assert not variant['report']['valid']
        assert [n['lane'] for n in variant['notes']]==[0,0]
        assert '结构冲突' in variant['selection']['message']
        assert client.get(base+'/export/osu').status_code==400
    finally:app.extensions['executor'].shutdown()


def test_original_model_cancel_during_postprocessing_stops_before_publication(tmp_path,monkeypatch):
    from manialab.upstream import MapperEngine

    app=create_app(ROOT,tmp_path);client=app.test_client();headers={'X-ManiaLab-Token':app.config['TOKEN']}
    fake_mapper_install(tmp_path)
    projects=Projects(tmp_path/'projects')
    imported=client.post('/api/import',data={'files':(io.BytesIO(b'placeholder'),'song.wav')},headers=headers).json
    pid=imported['id'];base=f'/api/projects/{pid}';project=projects.path(pid)
    (project/'preview.wav').write_bytes(b'placeholder')
    previous=projects.load(pid)
    postprocessing=threading.Event();release=threading.Event()

    def model(self,audio,output,prefs,progress,cancelled,seconds,tempo):
        output.mkdir(parents=True)
        target=output/'generated.osu'
        target.write_text(export_osu([Note(1000,4000,0),Note(2000,2000,1)],10,
            ROOT/'vendor/mapperatorinator/template.osu'),encoding='utf-8')
        return target

    def blocked_layout(*args):
        postprocessing.set()
        assert release.wait(5)
        return {}

    monkeypatch.setattr(MapperEngine,'generate',model)
    monkeypatch.setattr('manialab.upstream_routes.layout_metrics',blocked_layout)
    monkeypatch.setattr('manialab.upstream_routes.review_chart',
        lambda *args:pytest.fail('Cancelled model output continued into review'))
    try:
        response=client.post(base+'/engine-generate',json={'settings':{'keys':10}},headers=headers)
        assert response.status_code==200,response.json
        job=response.json
        assert postprocessing.wait(5)
        assert client.post('/api/jobs/'+job['id']+'/cancel',headers=headers).json=={'ok':True}
        release.set()
        result=wait_job(client,job)

        assert result['status']=='cancelled',result
        assert projects.load(pid)==previous
        assert not list((project/'files').glob('Mapperatorinator-*.osu'))
        assert not (project/'chart-workspaces').exists()
    finally:
        release.set()
        app.extensions['executor'].shutdown()


@pytest.mark.parametrize('has_previous_snapshot',[True,False])
def test_original_model_cancel_during_publication_rolls_back_files_and_workspace(tmp_path,monkeypatch,has_previous_snapshot):
    from manialab.chart_workspace import archive_active_chart, chart_workspace_snapshot_path
    from manialab.upstream import MapperEngine

    app=create_app(ROOT,tmp_path);client=app.test_client();headers={'X-ManiaLab-Token':app.config['TOKEN']}
    fake_mapper_install(tmp_path)
    projects=Projects(tmp_path/'projects')
    imported=client.post('/api/import',data={'files':(io.BytesIO(b'placeholder'),'song.wav')},headers=headers).json
    pid=imported['id'];base=f'/api/projects/{pid}';project=projects.path(pid)
    (project/'preview.wav').write_bytes(b'placeholder')
    previous=projects.load(pid)
    snapshot_path=chart_workspace_snapshot_path(project,None)
    previous_snapshot=b'{"original":null,"sentinel":"previous workspace"}'
    if has_previous_snapshot:
        snapshot_path.parent.mkdir()
        snapshot_path.write_bytes(previous_snapshot)
    archived=threading.Event();release=threading.Event()

    def model(self,audio,output,prefs,progress,cancelled,seconds,tempo):
        output.mkdir(parents=True)
        target=output/'generated.osu'
        target.write_text(export_osu([Note(1000,4000,0),Note(2000,2000,1)],10,
            ROOT/'vendor/mapperatorinator/template.osu'),encoding='utf-8')
        return target

    def blocked_archive(path,state):
        archive_active_chart(path,state)
        archived.set()
        assert release.wait(5)

    monkeypatch.setattr(MapperEngine,'generate',model)
    monkeypatch.setattr('manialab.upstream_routes.archive_active_chart',blocked_archive)
    try:
        response=client.post(base+'/engine-generate',json={'settings':{'keys':10}},headers=headers)
        assert response.status_code==200,response.json
        job=response.json
        assert archived.wait(5)
        assert len(list((project/'files').glob('Mapperatorinator-*.osu')))==1
        assert list((project/'chart-workspaces').glob('*.json'))
        if has_previous_snapshot:
            assert list((project/'chart-workspaces').glob('.*.rollback'))
        assert client.post('/api/jobs/'+job['id']+'/cancel',headers=headers).json=={'ok':True}
        release.set()
        result=wait_job(client,job)

        assert result['status']=='cancelled',result
        assert projects.load(pid)==previous
        assert not list((project/'files').glob('Mapperatorinator-*.osu'))
        assert not snapshot_path.with_suffix('.tmp').exists()
        if has_previous_snapshot:
            assert snapshot_path.read_bytes()==previous_snapshot
        else:
            assert not snapshot_path.parent.exists()
    finally:
        release.set()
        app.extensions['executor'].shutdown()


@pytest.mark.parametrize('has_previous_snapshot',[True,False])
def test_original_model_cancel_cleanup_continues_after_chart_unlink_failure(
        tmp_path,monkeypatch,caplog,has_previous_snapshot):
    from manialab.chart_workspace import archive_active_chart, chart_workspace_snapshot_path
    from manialab.upstream import MapperEngine

    app=create_app(ROOT,tmp_path);client=app.test_client();headers={'X-ManiaLab-Token':app.config['TOKEN']}
    fake_mapper_install(tmp_path)
    projects=app.extensions['projects']
    imported=client.post('/api/import',data={'files':(io.BytesIO(b'placeholder'),'song.wav')},headers=headers).json
    pid=imported['id'];base=f'/api/projects/{pid}';project=projects.path(pid)
    (project/'preview.wav').write_bytes(b'placeholder')
    snapshot_path=chart_workspace_snapshot_path(project,None)
    previous_snapshot=b'{"original":null,"sentinel":"previous workspace"}'
    if has_previous_snapshot:
        snapshot_path.parent.mkdir()
        snapshot_path.write_bytes(previous_snapshot)
    archived=threading.Event();release=threading.Event()

    def model(self,audio,output,prefs,progress,cancelled,seconds,tempo):
        output.mkdir(parents=True)
        target=output/'generated.osu'
        target.write_text(export_osu([Note(1000,4000,0),Note(2000,2000,1)],10,
            ROOT/'vendor/mapperatorinator/template.osu'),encoding='utf-8')
        return target

    def blocked_archive(path,state):
        archive_active_chart(path,state)
        archived.set()
        assert release.wait(5)

    original_unlink=Path.unlink
    denied_charts=[]
    def deny_generated_chart_unlink(path,*args,**kwargs):
        if path.name.startswith('Mapperatorinator-') and path.suffix=='.osu':
            denied_charts.append(path)
            raise PermissionError('synthetic generated chart unlink denied')
        return original_unlink(path,*args,**kwargs)

    monkeypatch.setattr(MapperEngine,'generate',model)
    monkeypatch.setattr('manialab.upstream_routes.archive_active_chart',blocked_archive)
    monkeypatch.setattr(Path,'unlink',deny_generated_chart_unlink)
    try:
        response=client.post(base+'/engine-generate',json={'settings':{'keys':10}},headers=headers)
        assert response.status_code==200,response.json
        job=response.json
        assert archived.wait(5)
        assert client.post('/api/jobs/'+job['id']+'/cancel',headers=headers).json=={'ok':True}
        release.set()
        result=wait_job(client,job)

        assert result['status']=='error',result
        assert result['cleanup_failed'] is True
        assert '资源清理失败' in result['message']
        assert 'synthetic generated chart unlink denied' in caplog.text
        assert len(denied_charts)==1 and denied_charts[0].is_file()
        if has_previous_snapshot:
            assert snapshot_path.read_bytes()==previous_snapshot
            assert not list(snapshot_path.parent.glob('*.rollback'))
        else:
            assert not snapshot_path.parent.exists()
    finally:
        release.set()
        app.extensions['executor'].shutdown()


def test_original_model_cancel_before_archive_ignores_missing_snapshot_directory(tmp_path,monkeypatch):
    from manialab.chart_workspace import chart_workspace_snapshot_path
    from manialab.upstream import MapperEngine

    app=create_app(ROOT,tmp_path);client=app.test_client();headers={'X-ManiaLab-Token':app.config['TOKEN']}
    fake_mapper_install(tmp_path)
    projects=app.extensions['projects']
    imported=client.post('/api/import',data={'files':(io.BytesIO(b'placeholder'),'song.wav')},headers=headers).json
    pid=imported['id'];base=f'/api/projects/{pid}';project=projects.path(pid)
    (project/'preview.wav').write_bytes(b'placeholder')
    previous=projects.load(pid)
    snapshot_path=chart_workspace_snapshot_path(project,None)
    assert not snapshot_path.parent.exists()
    chart_written=threading.Event();release=threading.Event()

    def model(self,audio,output,prefs,progress,cancelled,seconds,tempo):
        output.mkdir(parents=True)
        target=output/'generated.osu'
        target.write_text(export_osu([Note(1000,4000,0),Note(2000,2000,1)],10,
            ROOT/'vendor/mapperatorinator/template.osu'),encoding='utf-8')
        return target

    class BlockingWriter:
        def __init__(self,stream):
            self.stream=stream

        def __enter__(self):
            return self

        def __exit__(self,*args):
            return self.stream.__exit__(*args)

        def write(self,value):
            result=self.stream.write(value)
            chart_written.set()
            assert release.wait(5)
            return result

    original_open=Path.open
    def block_published_chart_write(path,*args,**kwargs):
        stream=original_open(path,*args,**kwargs)
        if path.name.startswith('Mapperatorinator-') and path.suffix=='.osu':
            return BlockingWriter(stream)
        return stream

    monkeypatch.setattr(MapperEngine,'generate',model)
    monkeypatch.setattr(Path,'open',block_published_chart_write)
    try:
        response=client.post(base+'/engine-generate',json={'settings':{'keys':10}},headers=headers)
        assert response.status_code==200,response.json
        job=response.json
        assert chart_written.wait(5)
        assert not snapshot_path.parent.exists()
        assert client.post('/api/jobs/'+job['id']+'/cancel',headers=headers).json=={'ok':True}
        release.set()
        result=wait_job(client,job)

        assert result['status']=='cancelled',result
        assert result.get('cleanup_failed') is not True
        assert projects.load(pid)==previous
        assert not list((project/'files').glob('Mapperatorinator-*.osu'))
        assert not snapshot_path.parent.exists()
    finally:
        release.set()
        app.extensions['executor'].shutdown()


@pytest.mark.parametrize('has_previous_snapshot',[True,False])
def test_original_model_save_failure_rolls_back_files_and_workspace(tmp_path,monkeypatch,has_previous_snapshot):
    from manialab.chart_workspace import chart_workspace_snapshot_path
    from manialab.upstream import MapperEngine

    app=create_app(ROOT,tmp_path);client=app.test_client();headers={'X-ManiaLab-Token':app.config['TOKEN']}
    fake_mapper_install(tmp_path)
    projects=app.extensions['projects']
    imported=client.post('/api/import',data={'files':(io.BytesIO(b'placeholder'),'song.wav')},headers=headers).json
    pid=imported['id'];base=f'/api/projects/{pid}';project=projects.path(pid)
    (project/'preview.wav').write_bytes(b'placeholder')
    previous=projects.load(pid)
    snapshot_path=chart_workspace_snapshot_path(project,None)
    previous_snapshot=b'{"original":null,"sentinel":"previous workspace"}'
    if has_previous_snapshot:
        snapshot_path.parent.mkdir()
        snapshot_path.write_bytes(previous_snapshot)

    def model(self,audio,output,prefs,progress,cancelled,seconds,tempo):
        output.mkdir(parents=True)
        target=output/'generated.osu'
        target.write_text(export_osu([Note(1000,4000,0),Note(2000,2000,1)],10,
            ROOT/'vendor/mapperatorinator/template.osu'),encoding='utf-8')
        return target

    save=projects.save
    def fail_published_state(state):
        if str(state.get('original','')).startswith('Mapperatorinator-'):
            raise OSError('simulated project state commit failure')
        return save(state)

    monkeypatch.setattr(MapperEngine,'generate',model)
    monkeypatch.setattr(projects,'save',fail_published_state)
    try:
        response=client.post(base+'/engine-generate',json={'settings':{'keys':10}},headers=headers)
        assert response.status_code==200,response.json
        result=wait_job(client,response.json)

        assert result['status']=='error',result
        assert projects.load(pid)==previous
        assert not list((project/'files').glob('Mapperatorinator-*.osu'))
        assert not snapshot_path.with_suffix('.tmp').exists()
        assert not list(snapshot_path.parent.glob('*.rollback'))
        if has_previous_snapshot:
            assert snapshot_path.read_bytes()==previous_snapshot
        else:
            assert not snapshot_path.parent.exists()
    finally:
        app.extensions['executor'].shutdown()


def test_archive_active_chart_removes_partial_snapshot_after_write_failure(tmp_path,monkeypatch):
    from manialab.chart_workspace import archive_active_chart, chart_workspace_snapshot_path

    project=tmp_path/'project'
    project.mkdir()
    state={'audio':'song.wav','original':None,'charts':[],'settings':{}}
    snapshot_path=chart_workspace_snapshot_path(project,None)
    temporary=snapshot_path.with_suffix('.tmp')
    write_text=Path.write_text

    def fail_after_partial_write(path,text,*args,**kwargs):
        if path==temporary:
            with path.open('w',encoding=kwargs.get('encoding','utf-8')) as output:
                output.write(text[:8])
                output.flush()
            raise OSError('simulated snapshot write failure')
        return write_text(path,text,*args,**kwargs)

    monkeypatch.setattr(Path,'write_text',fail_after_partial_write)
    with pytest.raises(OSError,match='simulated snapshot write failure'):
        archive_active_chart(project,state)

    assert not temporary.exists()
    assert not snapshot_path.exists()


def test_mapping_clock_changes_export_once_and_audition_uses_actual_chart_time(tmp_path):
    import numpy as np
    from scipy.io import wavfile
    app=create_app(ROOT,tmp_path);client=app.test_client();headers={'X-ManiaLab-Token':app.config['TOKEN']}
    projects=Projects(tmp_path/'projects')
    state=client.post('/api/import',data={'files':(io.BytesIO(b'placeholder'),'song.wav')},headers=headers).json
    pid=state['id'];base=f'/api/projects/{pid}'
    raw=json.loads(json.dumps([Note(115+i*500,865+i*500,source='pitch',id=str(i)).data() for i in range(12)]))
    state['analysis']=dict(revision=5,candidates=raw,timing_attacks=raw,tempo=dict(bpm=120,offset=115,division=4,stable=True))
    projects.save(state)
    payload=dict(mode='audio',audio_shift_ms=22.5,settings={'ln_ratio':1,'density':4})
    result=wait_job(client,client.post(base+'/generate',json=payload,headers=headers).json)
    assert result['status']=='done',result
    saved=client.get(base).json;v=saved['variants'][0]
    assert saved['analysis']['candidates']==raw
    assert v['offset']==93 and v['audio_shift_ms']==22.5
    assert all(n['start']%500==93 and n['end']-n['start']==750 for n in v['notes'])
    parsed=parse_osu(client.get(base+'/export/osu').data.decode())
    assert float(parsed['timing'][0].split(',')[0])==93
    wavfile.write(projects.path(pid)/'preview.wav',22050,np.zeros(22050*8,dtype=np.int16))
    response=client.post(base+'/audition',json={'mode':'chart','duration_ms':1000},headers=headers)
    rate,wave=wavfile.read(io.BytesIO(response.data))
    assert abs(np.flatnonzero(wave[:,0])[0]/rate*1000-93)<.1
    # Changing the convention cannot silently mix clocks in a local rewrite.
    failed=wait_job(client,client.post(base+'/generate',json=payload|{'audio_shift_ms':0,'region':[1000,3000]},headers=headers).json)
    assert failed['status']=='error' and 'Timing' in failed['message']
    app.extensions['executor'].shutdown()


def test_reference_style_preserves_audio_candidates_and_target_keys(tmp_path):
    app=create_app(ROOT,tmp_path);client=app.test_client();headers={'X-ManiaLab-Token':app.config['TOKEN']}
    state=client.post('/api/import',data={'files':(io.BytesIO(b'placeholder'),'song.wav')},headers=headers).json
    projects=Projects(tmp_path/'projects');state['settings']['keys']=10
    state['analysis']={'sentinel':'original evidence'};projects.save(state)
    notes=[Note(t,t+500 if i%2 else t,i%10) for i,t in enumerate(range(0,10000,250))]
    text=export_osu(notes,10,ROOT/'vendor/mapperatorinator/template.osu')
    response=client.post(f"/api/projects/{state['id']}/reference-style",data={'file':(io.BytesIO(text.encode()),'reference.osu'),
        'settings':json.dumps({'keys':6,'pitch_follow':.7})},headers=headers)
    assert response.status_code==200,response.json
    saved=response.json
    assert saved['audio']=='song.wav' and saved['analysis']==state['analysis']
    assert saved['settings']['keys']==6 and saved['settings']['ln_ratio']==.5
    assert saved['settings']['pitch_follow']==.7
    assert saved['reference_style']['keys']==10 and saved['reference_style']['objects']==40
    assert saved['variants']==[] and saved['original'] is None
    app.extensions['executor'].shutdown()


def test_audition_http_preserves_project_and_returns_aligned_wave(tmp_path):
    import numpy as np
    from scipy.io import wavfile
    app=create_app(ROOT,tmp_path);client=app.test_client();headers={'X-ManiaLab-Token':app.config['TOKEN']}
    state=client.post('/api/import',data={'files':(io.BytesIO(b'placeholder'),'song.wav')},headers=headers).json
    pid=state['id'];url=f'/api/projects/{pid}/audition'
    assert client.post(url,json={},headers=headers).status_code==400
    preview=tmp_path/'projects'/pid/'preview.wav'
    wavfile.write(preview,22050,np.zeros(22050*10,dtype=np.int16))
    before=(preview.parent/'state.json').read_bytes()
    result=client.post(url,json={'bpm':120,'offset':91,'start_ms':0,'duration_ms':5000},headers=headers)
    assert result.status_code==200 and result.mimetype=='audio/wav'
    sr,content=wavfile.read(io.BytesIO(result.data))
    assert sr==22050 and content.shape==(110250,2)
    first=np.flatnonzero(content[:,0])[0]
    assert abs(first/22.05-91)<.1
    assert (preview.parent/'state.json').read_bytes()==before
    assert client.post(url,json={'duration_ms':float('nan')},headers=headers).status_code==400
    assert client.post(url,json={'mode':'chart'},headers=headers).status_code==400
    app.extensions['executor'].shutdown()


@pytest.mark.parametrize('keys', range(6,11))
def test_reference_210_91_export_and_timing_guards(tmp_path,keys,monkeypatch):
    app=create_app(ROOT,tmp_path);client=app.test_client();headers={'X-ManiaLab-Token':app.config['TOKEN']}
    projects=Projects(tmp_path/'projects')
    state=client.post('/api/import',data={'files':(io.BytesIO(b'placeholder'),'song.wav')},headers=headers).json
    pid=state['id'];base=f'/api/projects/{pid}'
    tick=60000/210/4
    notes=[Note(round(91+i*4*tick+24),round(91+(i*4+3)*tick+31),source='pitch',id=str(i)) for i in range(100)]
    state['analysis']={'revision':3,'candidates':[n.data() for n in notes],
        'timing_attacks':[Note(round(91+i*4*tick),round(91+i*4*tick)).data() for i in range(100)],
        'tempo':{'bpm':210,'offset':118,'stable':False,'division':4}}
    projects.save(state)
    payload=dict(mode='audio',settings={'keys':keys,'ln_ratio':1,'min_ln':100},bpm=210,offset=91,division=4)
    result=wait_job(client,client.post(base+'/generate',json=payload,headers=headers).json)
    assert result['status']=='error' and 'Timing' in result['message']
    fitted=client.post(base+'/timing-refit',json=payload,headers=headers).json
    assert fitted['offset']==91
    payload.update(timing_confirmed=True)
    result=wait_job(client,client.post(base+'/generate',json=payload,headers=headers).json)
    assert result['status']=='done',result
    response=client.get(base+'/export/osu');assert response.status_code==200,response.json
    parsed=parse_osu(response.data.decode())
    assert parsed['keys']==keys and any(n.hold for n in parsed['notes'])
    assert any(line.startswith('91.000,285.71428571,') for line in parsed['timing'])
    for n in parsed['notes']:
        assert all(abs(t-(91+round((t-91)/tick)*tick))<=.501 for t in (n.start,n.end))
    result=wait_job(client,client.post(base+'/generate',json=payload|{'offset':118,'region':[0,5000]},headers=headers).json)
    assert result['status']=='error' and 'Timing 已改变' in result['message']
    stored=projects.load(pid);stored['variants'][0]['timing_revision']=2;projects.save(stored)
    assert client.get(base+'/export/osu').status_code==400
    assert client.get(base+'/export/json').status_code==200
    monkeypatch.setattr(app.extensions['analysis_services'],'analyze',lambda *args:state['analysis'])
    result=wait_job(client,client.post(base+'/analyze',json={},headers=headers).json)
    assert result['status']=='done'
    refreshed=projects.load(pid)
    assert refreshed['active']==-1 and len(refreshed['variants'])==1
    diagnosis=client.get(base+'/export/json')
    assert diagnosis.status_code==200 and diagnosis.json['variant'] is None
    assert diagnosis.json['analysis']['tempo']['offset']==118
    app.extensions['executor'].shutdown()
