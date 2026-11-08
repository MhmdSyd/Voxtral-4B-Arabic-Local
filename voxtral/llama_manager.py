"""Find, start, watch and stop llama-server."""
import asyncio
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from . import config

IS_WIN = sys.platform == "win32"
EXE = "llama-server.exe" if IS_WIN else "llama-server"

INSTALL_HINT = (
    "Install llama.cpp (build b11675 / v0.8.0 or newer), then press Retry.\n"
    "  Windows:  winget install llama.cpp   (then open a new terminal)\n"
    "  Or download a CUDA build from https://github.com/ggml-org/llama.cpp/releases\n"
    "  and set LLAMA_SERVER to the full path of llama-server.exe"
)


class SetupError(Exception):
    def __init__(self, message, hint=None, log=None):
        super().__init__(message)
        self.message, self.hint, self.log = message, hint, log


def check_version(binary: str):
    """Return a problem description if the build is clearly too old, else None."""
    try:
        r = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=30)
        text = (r.stdout or "") + (r.stderr or "")
    except Exception:  # noqa: BLE001
        return None
    m = re.search(r"version:\s*v?(\d+)\.(\d+)", text)
    if m and (int(m.group(1)), int(m.group(2))) < (0, 8):
        return f"This llama.cpp ({m.group(0)}) is too old for Voxtral Realtime. Version 0.8.0 or newer is needed."
    m = re.search(r"version:\s*b?(\d+)\b(?!\.)", text)
    if m and int(m.group(1)) < config.MIN_BUILD:
        return f"This llama.cpp build ({m.group(1)}) is too old for Voxtral Realtime. Build {config.MIN_BUILD} (v0.8.0) or newer is needed."
    return None


class LlamaServer:
    def __init__(self):
        self.proc = None
        self.external = False
        self.binary = None
        self.model = self.mmproj = None
        self._log_fh = None
        self.log_path = config.LOGS_DIR / "llama-server.log"

    # ---- discovery ---------------------------------------------------
    def find_binary(self):
        if config.LLAMA_SERVER and Path(config.LLAMA_SERVER).exists():
            return config.LLAMA_SERVER
        found = shutil.which("llama-server")
        if found:
            return found
        local = config.ROOT / "bin"
        if local.exists():
            for p in local.rglob(EXE):
                return str(p)
        if IS_WIN:  # winget installs may not be on PATH in an already-open terminal
            base = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet"
            link = base / "Links" / EXE
            if link.exists():
                return str(link)
            pk = base / "Packages"
            if pk.exists():
                for d in pk.glob("*llama*"):
                    for p in d.rglob(EXE):
                        return str(p)
        return None

    # ---- health ------------------------------------------------------
    def _probe(self) -> bool:
        try:
            with urllib.request.urlopen(config.LLAMA_URL + "/health", timeout=2) as r:
                return r.status == 200
        except (urllib.error.URLError, OSError):
            return False

    async def healthy(self) -> bool:
        return await asyncio.to_thread(self._probe)

    def running(self) -> bool:
        return self.external or (self.proc is not None and self.proc.poll() is None)

    # ---- lifecycle ---------------------------------------------------
    def start(self, binary: str, model: Path, mmproj: Path):
        self.binary, self.model, self.mmproj = binary, model, mmproj
        config.LOGS_DIR.mkdir(exist_ok=True)
        self._log_fh = open(self.log_path, "w", encoding="utf-8", errors="replace")
        cmd = [
            binary, "-m", str(model), "--mmproj", str(mmproj),
            "-ngl", "99",            # every layer on the GPU
            "-c", config.CTX_SIZE,
            "--temp", "0",           # the model card evaluates with temperature 0
            "--host", config.LLAMA_HOST, "--port", str(config.LLAMA_PORT),
        ]
        self._log_fh.write("$ " + " ".join(cmd) + "\n")
        self._log_fh.flush()
        kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if IS_WIN else {"start_new_session": True}
        self.proc = subprocess.Popen(cmd, stdout=self._log_fh, stderr=subprocess.STDOUT, **kw)

    async def wait_ready(self, timeout=300):
        waited = 0.0
        while waited < timeout:
            if self.proc.poll() is not None:
                raise SetupError(
                    f"llama-server stopped while loading the model (exit code {self.proc.returncode}).",
                    "Details are in logs/llama-server.log (last lines below).",
                    self.log_tail(),
                )
            if await self.healthy():
                return
            await asyncio.sleep(0.5)
            waited += 0.5
        raise SetupError("llama-server took too long to load the model.", None, self.log_tail())

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        if self._log_fh:
            try:
                self._log_fh.close()
            except Exception:  # noqa: BLE001
                pass
            self._log_fh = None
        self.proc = None

    def restart(self):
        """Start again with the same files (used to load the model back after an idle unload)."""
        self.start(self.binary, self.model, self.mmproj)

    # ---- log helpers -------------------------------------------------
    def log_text(self) -> str:
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def log_tail(self, n=18) -> str:
        return "\n".join(self.log_text().splitlines()[-n:])

    def gpu_info(self):
        """Best effort: pick the GPU name out of llama-server's startup log."""
        text = self.log_text()
        m = re.search(r"Device \d+:\s*([^\n,(]+)", text) or re.search(r"ggml_vulkan:\s*\d+\s*=\s*([^|(\n]+)", text)
        return m.group(1).strip() if m else None
