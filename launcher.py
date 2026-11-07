from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
from pathlib import Path
import socket
import sys
import threading
import webbrowser


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
    server.run()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
