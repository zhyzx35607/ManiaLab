from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
from pathlib import Path
import socket
import sys
import threading
import time
import webbrowser

from manialab.cancellation import remaining_until

SHUTDOWN_GRACE_SECONDS = 15


def _run_server(server, task_service):
    try:
        server.run()
    finally:
        deadline = time.monotonic() + SHUTDOWN_GRACE_SECONDS
        try:
            task_service.shutdown(wait=False, deadline=deadline)
        finally:
            try:
                server.close()
            finally:
                try:
                    dispatcher = getattr(server, "task_dispatcher", None)
                    if dispatcher is not None:
                        dispatcher.shutdown(cancel_pending=True, timeout=remaining_until(deadline))
                finally:
                    stopped = task_service.shutdown(wait=True, deadline=deadline)
                    if not stopped:
                        message = "Task service did not stop before the shutdown deadline"
                        logging.error(message)
                        raise ChildProcessError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--data-dir")
    args = parser.parse_args()
    resources = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
    root = Path(args.data_dir) if args.data_dir else Path(os.getenv("LOCALAPPDATA", Path.home())) / "ManiaLab"
    root.mkdir(parents=True, exist_ok=True)
    # Keep one writer per data directory. Reopening the app opens its existing UI.
    singleton = None
    if os.name == "nt":
        import msvcrt
        singleton = (root / "instance.lock").open("a+b")
        if os.fstat(singleton.fileno()).st_size == 0:
            singleton.write(b"0");singleton.flush()
        singleton.seek(0)
        try:
            msvcrt.locking(singleton.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            existing = root / "current-url.txt"
            if existing.exists() and not args.no_browser:
                webbrowser.open(existing.read_text(encoding="utf-8").strip())
            return
    if sys.stdout is None:
        sys.stdout = (root / "ManiaLab.log").open("a", encoding="utf-8", buffering=1)
    if sys.stderr is None:
        sys.stderr = sys.stdout
    # Frozen Python can ignore PYTHONUTF8 on English Windows. Redirected
    # stdout then uses cp1252 and even our startup message would abort launch.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    logging.basicConfig(filename=root / "ManiaLab.log", level=logging.INFO, encoding="utf-8")
    from manialab.server import create_app
    from waitress import create_server
    app = create_app(resources, root)
    server = create_server(app, host="127.0.0.1", port=args.port, threads=6)
    url = f"http://127.0.0.1:{server.effective_port}"
    (root / "current-url.txt").write_text(url, encoding="utf-8")
    print(f"ManiaLab: {url}\n项目数据: {root}\n关闭此窗口退出程序。", flush=True)
    if not args.no_browser:
        threading.Timer(.6, lambda: webbrowser.open(url)).start()
    _run_server(server, app.extensions["task_service"])


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
