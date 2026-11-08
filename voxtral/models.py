"""Download the GGUF files with progress and resume."""
import shutil
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from . import config


@dataclass
class ModelFile:
    name: str
    label: str
    url: str = ""  # default: the Hugging Face repo from config
    path: Path = field(init=False)
    total: int = 0
    downloaded: int = 0
    status: str = "wait"  # wait | run | done | error

    def __post_init__(self):
        self.path = config.MODELS_DIR / self.name
        if not self.url:
            self.url = f"{config.HF_ENDPOINT}/{config.HF_REPO}/resolve/main/{self.name}?download=true"


def default_files():
    return [
        ModelFile(config.MODEL_FILE, "Voxtral Arabic Q8_0 model"),
        ModelFile(config.MMPROJ_FILE, "Audio encoder (mmproj)"),
    ]


def _download_one(f: ModelFile, stop: threading.Event, attempts: int = 5):
    part = f.path.with_name(f.path.name + ".part")
    f.path.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for attempt in range(1, attempts + 1):
        if stop.is_set():
            raise InterruptedError("stopped")
        try:
            pos = part.stat().st_size if part.exists() else 0
            headers = {"User-Agent": "voxtral-local/1.0"}
            if pos:
                headers["Range"] = f"bytes={pos}-"
            with urllib.request.urlopen(urllib.request.Request(f.url, headers=headers), timeout=30) as r:
                resumed = pos > 0 and r.status == 206
                if resumed:
                    total = int(r.headers.get("Content-Range", "bytes 0-0/0").rsplit("/", 1)[-1])
                else:
                    pos = 0
                    total = int(r.headers.get("Content-Length", 0))
                f.total = total
                free = shutil.disk_usage(config.MODELS_DIR).free
                if total and free < (total - pos) + 200 * 1024 * 1024:
                    raise OSError(f"Not enough free disk space in {config.MODELS_DIR} (need about {(total - pos) // 2**20} MB).")
                f.downloaded = pos
                with open(part, "ab" if resumed else "wb") as out:
                    while True:
                        if stop.is_set():
                            raise InterruptedError("stopped")
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
                        f.downloaded += len(chunk)
            if total and part.stat().st_size != total:
                raise OSError("Download ended early (size mismatch).")
            part.replace(f.path)
            return
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 416 and part.exists():  # bad resume point, start over
                part.unlink()
        except (InterruptedError, OSError) as e:
            if isinstance(e, InterruptedError) or "disk space" in str(e):
                raise
            last_err = e
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(min(2 ** attempt, 15))
    raise RuntimeError(f"Could not download {f.name}: {last_err}")


def ensure_downloaded(files, stop: threading.Event):
    """Blocking. Runs in a background thread."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for f in files:
        if f.path.exists():
            f.total = f.downloaded = f.path.stat().st_size
            f.status = "done"
    for f in files:
        if f.status == "done":
            continue
        f.status = "run"
        try:
            _download_one(f, stop)
        except Exception:
            f.status = "error"
            raise
        f.status = "done"
