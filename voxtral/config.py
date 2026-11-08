"""All settings in one place. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("MODELS_DIR", ROOT / "models"))
LOGS_DIR = ROOT / "logs"
STATIC_DIR = ROOT / "static"
SESSIONS_DIR = ROOT / "sessions"  # saved recordings, one folder per browser tab

# Web app
APP_HOST = os.environ.get("HOST", "127.0.0.1")
APP_PORT = int(os.environ.get("PORT", "8000"))
OPEN_BROWSER = os.environ.get("OPEN_BROWSER", "1") == "1"

# Model files (Q8_0 + its audio encoder)
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
HF_REPO = os.environ.get("HF_REPO", "mradermacher/Voxtral-Mini-4B-Realtime-Arabic-GGUF")
MODEL_FILE = os.environ.get("MODEL_FILE", "Voxtral-Mini-4B-Realtime-Arabic.Q8_0.gguf")
MMPROJ_FILE = os.environ.get("MMPROJ_FILE", "Voxtral-Mini-4B-Realtime-Arabic.mmproj-Q8_0.gguf")

# llama-server (started and stopped by this app)
LLAMA_SERVER = os.environ.get("LLAMA_SERVER", "")  # full path, if not on PATH
LLAMA_HOST = "127.0.0.1"
LLAMA_PORT = int(os.environ.get("LLAMA_PORT", "8080"))
CTX_SIZE = os.environ.get("CTX_SIZE", "16384")  # ~12.5 tokens per audio second -> about 20 min per session
MIN_BUILD = 11675  # first llama.cpp build with Voxtral Realtime (v0.8.0)

LLAMA_URL = f"http://{LLAMA_HOST}:{LLAMA_PORT}"
REALTIME_PATH = os.environ.get("LLAMA_REALTIME_PATH", "/v1/realtime?intent=transcription")
DEBUG = os.environ.get("DEBUG", "") == "1"
IDLE_TIMEOUT = float(os.environ.get("COMMIT_IDLE_TIMEOUT", "10"))

# Tab sessions: recordings are deleted this many seconds after the browser tab is closed
# (the delay lets a page refresh keep its recordings).
SESSION_GRACE = float(os.environ.get("SESSION_GRACE_SECONDS", "30"))
MIN_RECORDING_SECONDS = 0.5

# The model is removed from the GPU after this many seconds without a request (0 = never).
IDLE_UNLOAD_SECONDS = float(os.environ.get("IDLE_UNLOAD_SECONDS", "120"))

# Speaker separation (optional, CPU, sherpa-onnx)
DIAR_DIR = MODELS_DIR / "diarization"
DIAR_SEG_URL = os.environ.get(
    "DIAR_SEG_URL",
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
)
DIAR_EMB_URL = os.environ.get(
    "DIAR_EMB_URL",
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34.onnx",
)
DIAR_THRESHOLD = float(os.environ.get("DIAR_THRESHOLD", "0.5"))  # lower = more speakers


def ws_url() -> str:
    return LLAMA_URL.replace("http", "ws", 1) + REALTIME_PATH
