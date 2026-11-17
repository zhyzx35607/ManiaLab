"""Cheap status polling with identity verification when a model file changes."""
import hashlib
from pathlib import Path
import threading
import time


class VerifiedModel:
    def __init__(self, path, byte_count, sha256):
        self.path = Path(path)
        self.byte_count = byte_count
        self.sha256 = sha256
        self._signature = None
        self._state = 'missing'
        self._lock = threading.Lock()
        self._checked_at = 0.

    def state(self, *, refresh=False):
        with self._lock:
            try:
                stat = self.path.stat()
                signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                if refresh or signature != self._signature or time.monotonic() - self._checked_at >= 30:
                    if not self.path.is_file() or stat.st_size != self.byte_count:
                        status = 'mismatched'
                    else:
                        with self.path.open('rb') as stream:
                            status = 'verified' if hashlib.file_digest(stream, 'sha256').hexdigest() == self.sha256 else 'mismatched'
                    # A concurrently replaced model must be checked on the next poll.
                    after = self.path.stat()
                    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) != signature:
                        self._signature = None
                        return 'changed'
                    self._signature, self._state = signature, status
                    self._checked_at = time.monotonic()
                return self._state
            except FileNotFoundError:
                self._signature = None
                return 'missing'
            except OSError:
                self._signature = None
                return 'unreadable'
