"""Optional API planner: only bounded preferences; never creates note timestamps."""
import asyncio
import base64
import json
import socket
import ssl
import threading
import urllib.parse
import urllib.request

from .arrange_config import parameters

MAX_RESPONSE_BYTES = 128 * 1024
REQUEST_TIMEOUT_SECONDS = 90
HEADER_LIMIT_BYTES = 64 * 1024

ALLOWED = {"density", "ln_ratio", "min_ln", "max_held", "chord_size", "focus", "pitch_follow", "balance", "jack_avoid", "release_gap",
           "phrase_ln", "drums_weight", "bass_weight", "vocals_weight", "other_weight", "guitar_weight", "piano_weight",
           "ln_coordination", "ln_release", "ln_inverse"}


def validate_plan(value, segments):
    if not isinstance(value, dict) or set(value) - {"explanation", "preferences", "segments"}:
        raise ValueError("API 必须返回 explanation/preferences/segments JSON")
    prefs = value.get("preferences", {})
    if not isinstance(prefs, dict) or set(prefs) - ALLOWED:
        raise ValueError("API 返回了未允许的全局参数")
    parameters(prefs)
    result = []
    entries = value.get("segments", [])
    if not isinstance(entries, list) or len(entries) > len(segments):
        raise ValueError("API 返回的段落数无效")
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"index", "preferences"}:
            raise ValueError("API 段落格式无效")
        index, local = entry["index"], entry["preferences"]
        if type(index) is not int or index < 0 or index >= len(segments) or index in seen:
            raise ValueError("API 段落索引无效或重复")
        if not isinstance(local, dict) or set(local) - ALLOWED:
            raise ValueError("API 返回了未允许的段落参数")
        parameters(local)
        seen.add(index)
        result.append({"start_ms": segments[index]["start_ms"], "end_ms": segments[index]["end_ms"], "preferences": local})
    explanation = value.get("explanation", "")
    if not isinstance(explanation, str):
        raise ValueError("API explanation 必须是文本")
    return {"preferences": prefs, "segments": result, "explanation": explanation[:3000]}


class HTTPStatusError(Exception):
    def __init__(self, status):
        self.status = status


class _BoundedResolver:
    """Resolve one host at a time outside asyncio's shutdown-waited executor."""

    def __init__(self, getaddrinfo=None):
        self._getaddrinfo = getaddrinfo or socket.getaddrinfo
        self._condition = threading.Condition()
        self._pending = None
        self._busy = False
        self._thread = None

    async def resolve(self, host, port):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        with self._condition:
            if self._busy:
                raise OSError("DNS resolver is busy")
            self._busy = True
            self._pending = (loop, future, host, port)
            if self._thread is None:
                try:
                    self._thread = threading.Thread(
                        target=self._run,
                        name="manialab-planner-resolver",
                        daemon=True,
                    )
                    self._thread.start()
                except BaseException:
                    self._busy = False
                    self._pending = None
                    self._thread = None
                    raise
            self._condition.notify()
        return await future

    def _run(self):
        while True:
            with self._condition:
                while self._pending is None:
                    self._condition.wait()
                loop, future, host, port = self._pending
                self._pending = None
            try:
                result = self._getaddrinfo(
                    host, port, socket.AF_UNSPEC, socket.SOCK_STREAM,
                    socket.IPPROTO_TCP,
                )
                error = None
            except BaseException as exc:
                result = None
                error = exc
            with self._condition:
                self._busy = False
            try:
                loop.call_soon_threadsafe(self._deliver, future, result, error)
            except RuntimeError:
                # The request's event loop may have closed after cancellation.
                pass

    @staticmethod
    def _deliver(future, result, error):
        if future.done():
            return
        if error is not None:
            future.set_exception(error)
        else:
            future.set_result(result)


_RESOLVER = None
_RESOLVER_LOCK = threading.Lock()


def _resolver():
    global _RESOLVER
    with _RESOLVER_LOCK:
        if _RESOLVER is None:
            _RESOLVER = _BoundedResolver()
        return _RESOLVER


async def _open_resolved_connection(host, port, *, ssl_context, server_hostname, limit):
    loop = asyncio.get_running_loop()
    lookup_host = host.encode("idna").decode("ascii")
    addresses = await _resolver().resolve(lookup_host, port)
    last_error = None
    for family, socktype, proto, _, sockaddr in addresses:
        connection = socket.socket(family, socktype, proto)
        connection.setblocking(False)
        try:
            await loop.sock_connect(connection, sockaddr)
        except asyncio.CancelledError:
            connection.close()
            raise
        except OSError as exc:
            connection.close()
            last_error = exc
            continue
        except BaseException:
            connection.close()
            raise
        try:
            reader, writer = await asyncio.open_connection(
                sock=connection,
                ssl=ssl_context,
                server_hostname=server_hostname,
                limit=limit,
            )
        except BaseException:
            connection.close()
            raise
        return reader, writer
    if last_error is not None:
        raise last_error
    raise OSError("DNS resolver returned no stream addresses")


def _host_header(host, port, scheme):
    host = host.encode("idna").decode("ascii")
    if ":" in host:
        host = f"[{host}]"
    default_port = 443 if scheme == "https" else 80
    return host if port == default_port else f"{host}:{port}"


def _proxy_for(parsed):
    if urllib.request.proxy_bypass(parsed.hostname):
        return None
    proxies = urllib.request.getproxies()
    value = proxies.get(parsed.scheme) or proxies.get("all")
    if not value:
        return None
    if "://" not in value:
        value = "http://" + value
    proxy = urllib.parse.urlsplit(value)
    if proxy.scheme not in ("http", "https") or not proxy.hostname:
        raise ValueError("API 代理地址格式无效")
    try:
        port = proxy.port
    except ValueError:
        raise ValueError("API 代理地址格式无效") from None
    if port == 0:
        raise ValueError("API 代理地址格式无效")
    if port is None:
        port = 443 if proxy.scheme == "https" else 80
    authorization = None
    if proxy.username is not None or proxy.password is not None:
        credentials = base64.b64encode(
            f"{urllib.parse.unquote(proxy.username or '')}:{urllib.parse.unquote(proxy.password or '')}".encode()
        ).decode("ascii")
        authorization = "Basic " + credentials
    return proxy, port, authorization


async def _read_headers(reader):
    try:
        raw = await reader.readuntil(b"\r\n\r\n")
    except asyncio.LimitOverrunError as error:
        raise ValueError("API 响应头过大") from error
    if len(raw) > HEADER_LIMIT_BYTES:
        raise ValueError("API 响应头过大")
    lines = raw[:-4].split(b"\r\n")
    status_line = lines[0].split(b" ", 2)
    if len(status_line) < 2 or not status_line[0].startswith(b"HTTP/"):
        raise ValueError("API 响应格式无效")
    try:
        status = int(status_line[1])
    except ValueError:
        raise ValueError("API 响应状态无效") from None
    headers = {}
    for line in lines[1:]:
        if b":" not in line:
            raise ValueError("API 响应头格式无效")
        name, value = line.split(b":", 1)
        try:
            key = name.decode("ascii").lower()
            item = value.decode("latin-1").strip()
        except UnicodeError:
            raise ValueError("API 响应头格式无效") from None
        headers.setdefault(key, []).append(item)
    return status, headers


async def _read_final_headers(reader):
    for _ in range(8):
        status, headers = await _read_headers(reader)
        if 100 <= status < 200:
            if status == 101:
                raise ValueError("API 不支持协议升级")
            continue
        return status, headers
    raise ValueError("API 临时响应次数过多")


async def _read_chunked(reader):
    body = bytearray()
    while True:
        line = await reader.readline()
        if not line.endswith(b"\r\n"):
            raise ValueError("API 分块响应格式无效")
        try:
            size = int(line[:-2].split(b";", 1)[0], 16)
        except ValueError:
            raise ValueError("API 分块响应格式无效") from None
        if size < 0 or len(body) + size > MAX_RESPONSE_BYTES:
            raise ValueError("API 响应过大")
        if size == 0:
            trailer_size = 0
            while True:
                trailer = await reader.readline()
                trailer_size += len(trailer)
                if trailer_size > HEADER_LIMIT_BYTES:
                    raise ValueError("API 响应尾部过大")
                if trailer in (b"\r\n", b"\n", b""):
                    return bytes(body)
        body.extend(await reader.readexactly(size))
        if await reader.readexactly(2) != b"\r\n":
            raise ValueError("API 分块响应格式无效")


async def _read_body(reader, headers):
    transfer_encoding = [
        item.strip().lower()
        for value in headers.get("transfer-encoding", [])
        for item in value.split(",")
    ]
    lengths = headers.get("content-length", [])
    if transfer_encoding and lengths:
        raise ValueError("API 响应长度格式冲突")
    if transfer_encoding:
        if transfer_encoding != ["chunked"]:
            raise ValueError("API 响应编码不受支持")
        return await _read_chunked(reader)
    if lengths:
        if len(set(lengths)) != 1:
            raise ValueError("API 响应长度格式冲突")
        try:
            length = int(lengths[0])
        except ValueError:
            raise ValueError("API 响应长度格式无效") from None
        if length < 0:
            raise ValueError("API 响应长度格式无效")
        if length > MAX_RESPONSE_BYTES:
            raise ValueError("API 响应过大")
        return await reader.readexactly(length)

    body = bytearray()
    while len(body) <= MAX_RESPONSE_BYTES:
        chunk = await reader.read(min(16 * 1024, MAX_RESPONSE_BYTES + 1 - len(body)))
        if not chunk:
            return bytes(body)
        body.extend(chunk)
    raise ValueError("API 响应过大")


async def _post_json(url, body, headers):
    parsed = urllib.parse.urlsplit(url)
    host = parsed.hostname.encode("idna").decode("ascii")
    port = parsed.port if parsed.port is not None else (443 if parsed.scheme == "https" else 80)
    proxy_info = _proxy_for(parsed)
    proxy = proxy_info[0] if proxy_info else None
    proxy_port = proxy_info[1] if proxy_info else None
    proxy_authorization = proxy_info[2] if proxy_info else None

    request = urllib.request.Request(url, data=body, headers=headers)
    writer = None
    try:
        connect_host = proxy.hostname if proxy else host
        connect_port = proxy_port if proxy else port
        proxy_tls = bool(proxy and proxy.scheme == "https")
        context = ssl.create_default_context() if proxy_tls or parsed.scheme == "https" else None
        reader, writer = await _open_resolved_connection(
            connect_host,
            connect_port,
            ssl_context=context if proxy_tls or (not proxy and parsed.scheme == "https") else None,
            server_hostname=connect_host if proxy_tls else (host if not proxy and parsed.scheme == "https" else None),
            limit=HEADER_LIMIT_BYTES,
        )

        if proxy and parsed.scheme == "https":
            authority = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
            connect_headers = [("Host", authority), ("Connection", "keep-alive")]
            if proxy_authorization:
                connect_headers.append(("Proxy-Authorization", proxy_authorization))
            writer.write(
                (f"CONNECT {authority} HTTP/1.1\r\n" +
                 "".join(f"{name}: {value}\r\n" for name, value in connect_headers) +
                 "\r\n").encode("latin-1")
            )
            await writer.drain()
            status, _ = await _read_final_headers(reader)
            if status != 200:
                raise ValueError(f"API 代理连接失败（HTTP {status}）")
            await writer.start_tls(ssl.create_default_context(), server_hostname=host)

        target = url if proxy and parsed.scheme == "http" else request.selector
        target = urllib.parse.quote(target, safe="/%:@!$&'()*+,;=-._~")
        request_headers = {
            "Host": _host_header(host, port, parsed.scheme),
            "Content-Length": str(len(body)),
            "Connection": "close",
        }
        request_headers.update(request.headers)
        if proxy and parsed.scheme == "http" and proxy_authorization:
            request_headers["Proxy-Authorization"] = proxy_authorization
        if any("\r" in value or "\n" in value for value in request_headers.values()):
            raise ValueError("API 请求头格式无效")
        encoded_headers = "".join(
            f"{name}: {value}\r\n" for name, value in request_headers.items()
        )
        writer.write(f"POST {target} HTTP/1.1\r\n{encoded_headers}\r\n".encode("latin-1") + body)
        await writer.drain()
        status, response_headers = await _read_final_headers(reader)
        if 300 <= status < 400:
            raise ValueError("API 地址返回重定向；请填写最终接口地址")
        if status >= 400:
            raise HTTPStatusError(status)
        return await _read_body(reader, response_headers)
    finally:
        if writer is not None:
            writer.close()
            try:
                await asyncio.wait_for(writer.wait_closed(), timeout=0.2)
            except (asyncio.CancelledError, OSError, ssl.SSLError, asyncio.TimeoutError):
                pass


async def _cancel_request(request):
    request.cancel()
    try:
        await request
    except asyncio.CancelledError:
        pass
    except Exception:
        pass


async def _request_cancellable(url, body, headers, cancelled):
    request = asyncio.create_task(_post_json(url, body, headers))
    loop = asyncio.get_running_loop()
    deadline = loop.time() + REQUEST_TIMEOUT_SECONDS
    try:
        while True:
            if cancelled is not None and cancelled():
                await _cancel_request(request)
                raise InterruptedError("已取消，原有谱面已保留")
            remaining = deadline - loop.time()
            if remaining <= 0:
                await _cancel_request(request)
                if cancelled is not None and cancelled():
                    raise InterruptedError("已取消，原有谱面已保留")
                raise asyncio.TimeoutError
            done, _ = await asyncio.wait({request}, timeout=min(0.05, remaining))
            if done:
                if cancelled is not None and cancelled():
                    await _cancel_request(request)
                    raise InterruptedError("已取消，原有谱面已保留")
                return request.result()
    finally:
        if not request.done():
            await _cancel_request(request)


def plan(endpoint, model, key, instruction, segments, current, review=None, cancelled=None):
    if isinstance(endpoint, str) and any(ord(char) < 0x20 or ord(char) == 0x7f for char in endpoint):
        raise ValueError("API 地址格式无效")
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.username or parsed.password or parsed.query or parsed.fragment or not parsed.hostname:
        raise ValueError("API 地址格式无效")
    try:
        port = parsed.port
    except ValueError:
        raise ValueError("API 地址格式无效") from None
    if port == 0:
        raise ValueError("API 地址格式无效")
    if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in ("localhost", "127.0.0.1", "::1")):
        raise ValueError("API 地址必须为 HTTPS（本地服务可用 HTTP）")
    if not model.strip() or not instruction.strip():
        raise ValueError("请填写模型名和编排要求")
    prompt = ("你是 osu!mania 6–10K 编排参数助手。你没有听到音频，只能参考下列检测统计。"
              "不能声称听到了具体乐器，不允许创造音符、时间或改键数。返回 JSON 对象，不要 Markdown。"
              "结构: {explanation:中文说明,preferences:{参数},segments:[{index:已知段落索引,preferences:{参数}}]}。"
              "允许参数: density 1..24, ln_ratio 0..1, min_ln 100..2000毫秒, max_held 1..10, chord_size 1..10,"
              "focus balanced/pitched/transient, pitch_follow/balance/jack_avoid 0..1, release_gap 0..250毫秒。"
              "ln_coordination 0..1（低值分手配合，高值同手长按中打短键），ln_release 0..1（高值减少同手不同步放手），"
              "ln_inverse 0..1（在已有LN与下一音之间偏好短间隙接续；不延长音尾）。三项只用于连续定轨。"
              "phrase_ln 0..1 控制有连续人声证据的跨音高长条；drums_weight/bass_weight/vocals_weight/other_weight/guitar_weight/piano_weight 0..2 控制已分离声部取舍（0完全不采，1默认）。"
              "只有提供voices统计的段落才分析过分离声部，否则声部权重不起作用；不能把模型估计当作真实乐器标注。"
              "如有已有候选的review，它定位短放键、快速同轨和同手LN配合密集位置；结合用户目标调整相应段落，不能把这些提示都当错误。"
              "ln_ratio是目标LN物件数占比；程序在持续音证据和占轨约束下尽量达到，不能通过删短键或虚构音尾凑比例。不够证据时应在说明中指出。")
    payload = {"model": model.strip(), "messages": [{"role": "system", "content": prompt},
               {"role": "user", "content": json.dumps({"request": instruction[:6000], "current": current, "segments": segments, "review": review}, ensure_ascii=False)}],
               "temperature": .3, "max_tokens": 2500}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    url = endpoint.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    if cancelled is not None and cancelled():
        raise InterruptedError("已取消，原有谱面已保留")
    try:
        body = asyncio.run(_request_cancellable(
            url, json.dumps(payload).encode(), headers, cancelled,
        ))
    except InterruptedError:
        raise
    except HTTPStatusError as error:
        if cancelled is not None and cancelled():
            raise InterruptedError("已取消，原有谱面已保留") from None
        raise ValueError(f"API 请求失败（HTTP {error.status}）；检查地址、模型、额度和密钥") from None
    except (OSError, ssl.SSLError, asyncio.IncompleteReadError, asyncio.LimitOverrunError, asyncio.TimeoutError):
        if cancelled is not None and cancelled():
            raise InterruptedError("已取消，原有谱面已保留") from None
        raise ValueError("API 连接失败或超时") from None
    try:
        result = json.loads(body)["choices"][0]["message"]["content"].strip()
        if result.startswith("```"):
            result = result.split("\n", 1)[1].rsplit("```", 1)[0]
        return validate_plan(json.loads(result), segments)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        raise ValueError("API 未返回可用的 JSON 编排参数") from None
