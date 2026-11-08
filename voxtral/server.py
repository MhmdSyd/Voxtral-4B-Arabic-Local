"""FastAPI app: orchestrates setup (download + llama-server), serves the UI and the API."""
import asyncio
import os
import tempfile
import threading
import time
import uuid
import wave
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from . import config
from .diarize import Diarizer, make_turns, slice_for
from .llama_manager import INSTALL_HINT, LlamaServer, SetupError, check_version
from .models import default_files, ensure_downloaded
from .realtime import BusyError, Realtime
from .sessions import SessionStore, valid_rid, valid_sid

llama = LlamaServer()
realtime = Realtime()
sessions = SessionStore()
diarizer = Diarizer()
stop_event = threading.Event()
model_lock = asyncio.Lock()  # one load / unload at a time
_boot_task = _idle_task = None
jobs: dict = {}


class Status:
    def __init__(self):
        self.reset()

    def reset(self):
        # checking | downloading | starting | ready | error   (first start)
        # unloading | sleeping | waking                        (model taken off the GPU while idle)
        self.state = "checking"
        self.message = "Checking your setup..."
        self.error = self.hint = self.log = self.gpu = None
        self.files = default_files()

    def set(self, state, message):
        self.state, self.message = state, message

    def fail(self, message, hint=None, log=None):
        self.state, self.message = "error", message
        self.error, self.hint, self.log = message, hint, log

    def to_dict(self):
        return {
            "state": self.state,
            "message": self.message,
            "error": self.error,
            "hint": self.hint,
            "log": self.log,
            "gpu": self.gpu,
            "model": "llama.cpp · Q8_0",
            "idle_unload_seconds": config.IDLE_UNLOAD_SECONDS,
            "speakers": diarizer.to_dict(),
            "files": [
                {"name": f.name, "label": f.label, "total": f.total, "downloaded": f.downloaded, "status": f.status}
                for f in self.files
            ],
        }


status = Status()


async def _in_daemon_thread(fn, *args):
    """Run blocking work in a daemon thread so Ctrl+C never waits for a download."""
    loop = asyncio.get_running_loop()
    fut = loop.create_future()

    def work():
        try:
            res = fn(*args)
            loop.call_soon_threadsafe(lambda: fut.cancelled() or fut.set_result(res))
        except BaseException as e:  # noqa: BLE001
            loop.call_soon_threadsafe(lambda: fut.cancelled() or fut.set_exception(e))

    threading.Thread(target=work, daemon=True).start()
    return await fut


async def bootstrap():
    """check -> download -> start llama-server -> ready."""
    status.reset()
    try:
        if await llama.healthy():  # a llama-server is already running on the port: use it
            llama.external = True
            status.set("ready", "Using the llama-server that is already running.")
            realtime.touch()
            return

        status.set("checking", "Looking for llama-server...")
        binary = llama.find_binary()
        if not binary:
            raise SetupError("llama-server was not found on this computer.", INSTALL_HINT)
        problem = await asyncio.to_thread(check_version, binary)
        if problem:
            raise SetupError(problem, INSTALL_HINT)

        status.set("downloading", "Downloading the model (first start only)...")
        await _in_daemon_thread(ensure_downloaded, status.files, stop_event)

        status.set("starting", "Starting llama-server on the GPU...")
        llama.start(binary, status.files[0].path, status.files[1].path)
        await llama.wait_ready()

        status.gpu = llama.gpu_info()
        status.set("ready", "Ready")
        realtime.touch()
    except asyncio.CancelledError:
        raise
    except SetupError as e:
        status.fail(e.message, e.hint, e.log)
    except InterruptedError:
        pass
    except Exception as e:  # noqa: BLE001
        status.fail(f"{type(e).__name__}: {e}", None, llama.log_tail() or None)


# ---- model on the GPU only while it is needed ---------------------------------
USABLE_STATES = ("ready", "unloading", "sleeping", "waking")


async def _load_model():
    """Start llama-server again after an idle unload. Caller holds model_lock."""
    status.set("waking", "Loading the model onto the GPU...")
    try:
        llama.restart()
        await llama.wait_ready()
        status.set("ready", "Ready")
    except SetupError as e:
        status.fail(e.message, e.hint, e.log)
    except Exception as e:  # noqa: BLE001
        status.fail(f"{type(e).__name__}: {e}", None, llama.log_tail() or None)


async def ensure_ready() -> bool:
    """True when the model is loaded and usable. Loads it first if it was unloaded for being idle."""
    realtime.touch()  # a request just arrived: restart the idle countdown
    if status.state == "ready":
        return True
    if status.state in USABLE_STATES:
        async with model_lock:
            if status.state == "sleeping":
                await _load_model()
        realtime.touch()
        return status.state == "ready"
    return False


async def _unload_model():
    async with model_lock:
        # re-check under the lock: a request may have arrived while we waited for it
        if status.state != "ready" or realtime.busy.locked() or _jobs_running():
            return
        if time.monotonic() - realtime.last_used < config.IDLE_UNLOAD_SECONDS:
            return
        status.set("unloading", "Removing the model from the GPU...")
        await asyncio.to_thread(llama.stop)  # the process exit is what frees the GPU memory
        status.set("sleeping", "Model unloaded to free the GPU. It loads again when needed.")


async def _idle_watcher():
    while True:
        await asyncio.sleep(5)
        if config.IDLE_UNLOAD_SECONDS <= 0 or status.state != "ready" or llama.external or not llama.running():
            continue  # (a llama-server we did not start is not ours to stop)
        if realtime.busy.locked() or _jobs_running():
            continue
        if time.monotonic() - realtime.last_used >= config.IDLE_UNLOAD_SECONDS:
            await _unload_model()


def _jobs_running() -> bool:
    return any(j.state == "running" for j in jobs.values())


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _boot_task, _idle_task
    sessions.wipe_all()  # leftovers of a previous run
    sessions.on_expire.append(_cancel_jobs_of)
    threading.Thread(target=diarizer.prepare, args=(stop_event,), daemon=True).start()
    _boot_task = asyncio.create_task(bootstrap())
    _idle_task = asyncio.create_task(_idle_watcher())
    yield
    stop_event.set()
    _boot_task.cancel()
    _idle_task.cancel()
    llama.stop()
    shutil_rmtree(config.SESSIONS_DIR)


def shutil_rmtree(path):
    import shutil

    shutil.rmtree(path, ignore_errors=True)


app = FastAPI(lifespan=lifespan)


def decode_audio(path: str) -> np.ndarray:
    """Any common audio file -> 16 kHz mono float32."""
    from mistral_common.tokens.tokenizers.audio import Audio

    audio = Audio.from_file(path, strict=False)
    audio.resample(16000)
    return np.asarray(audio.audio_array, dtype=np.float32)


@app.get("/api/status")
def get_status():
    # If llama-server died after we were ready, say so.
    if status.state == "ready" and not llama.external and llama.proc is not None and llama.proc.poll() is not None:
        status.fail("llama-server stopped unexpectedly.", "Press Retry to start it again.", llama.log_tail())
    return status.to_dict()


@app.post("/api/speakers/retry")
def speakers_retry():
    if diarizer.state in ("unavailable", "idle"):
        threading.Thread(target=diarizer.prepare, args=(stop_event,), daemon=True).start()
    return {"ok": True}


@app.post("/api/retry")
async def retry():
    global _boot_task
    if status.state == "error" and (_boot_task is None or _boot_task.done()):
        llama.stop()
        llama.proc, llama.external = None, False
        _boot_task = asyncio.create_task(bootstrap())
    return {"ok": True}


@app.websocket("/ws")
async def live_transcription(ws: WebSocket, sid: str = ""):
    await ws.accept()
    if status.state in ("sleeping", "unloading", "waking"):
        await ws.send_json({"type": "loading"})
    if not await ensure_ready():
        msg = status.error if status.state == "error" and status.error else "The model is not ready yet."
        await ws.send_json({"type": "error", "message": msg})
        await ws.close()
        return
    await realtime.live(ws, sessions.get(sid) if valid_sid(sid) else None)


@app.websocket("/ws/session")
async def tab_session(ws: WebSocket, sid: str = ""):
    """One per browser tab. While it is open the tab's recordings are kept; when it closes they expire."""
    if not valid_sid(sid):
        await ws.close(code=1008)
        return
    await ws.accept()
    session = sessions.attach(sid, ws)
    try:
        await ws.send_json(session.to_dict())
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        sessions.detach(sid, ws)


# ---- saved recordings -----------------------------------------------------------
def _recording_or_404(sid: str, rid: str):
    session = sessions.find(sid)
    rec = session.recordings.get(rid) if session and valid_rid(rid) else None
    if rec is None:
        raise HTTPException(404, "This recording is gone (it expires when its tab is closed).")
    return session, rec


@app.get("/api/sessions/{sid}/recordings/{rid}/audio")
def recording_audio(sid: str, rid: str, download: int = 0):
    _, rec = _recording_or_404(sid, rid)
    if download:
        return FileResponse(rec.path, media_type="audio/wav", filename=f"{rec.name.lower().replace(' ', '-')}.wav")
    return FileResponse(rec.path, media_type="audio/wav")


@app.delete("/api/sessions/{sid}/recordings/{rid}")
async def delete_recording(sid: str, rid: str):
    session, _ = _recording_or_404(sid, rid)
    session.delete_recording(rid)
    await session.push()
    return Response(status_code=204)


# ---- speaker separation jobs -------------------------------------------------------
class Job:
    def __init__(self, sid=None):
        self.id = uuid.uuid4().hex[:12]
        self.sid = sid
        self.state = "running"  # running | done | error
        self.step = "diarizing"  # diarizing | loading | transcribing
        self.done = self.total = 0
        self.pct = 0
        self.result = None
        self.error = None
        self.cancel = threading.Event()

    def to_dict(self):
        return {
            "state": self.state, "step": self.step, "done": self.done, "total": self.total,
            "pct": self.pct, "result": self.result, "error": self.error,
        }


def _cancel_jobs_of(sid):
    for j in jobs.values():
        if j.sid == sid:
            j.cancel.set()


async def _speaker_job(job: Job, samples: np.ndarray):
    try:
        seconds = len(samples) / 16000

        def progress(done, total):
            job.pct = int(done * 100 / total) if total else 0
            return job.cancel.is_set()

        async with realtime.hold():  # the model is ours for the whole job
            segs = await asyncio.to_thread(diarizer.run, samples, progress)
            if job.cancel.is_set():
                raise InterruptedError("cancelled")
            turns = make_turns(segs, seconds)
            job.step, job.total = "loading", len(turns)
            if turns and not await ensure_ready():
                raise RuntimeError(status.error or "The model is not ready.")
            job.step = "transcribing"
            tail = np.zeros(int(0.5 * 16000), dtype=np.float32)  # silence so the last word is flushed
            for t in turns:
                if job.cancel.is_set():
                    raise InterruptedError("cancelled")
                audio = np.concatenate([slice_for(samples, t), tail])
                try:
                    t["text"] = (await realtime._transcribe(audio, idle=5)).strip()
                except Exception as e:  # noqa: BLE001  one bad turn should not lose the others
                    t["text"] = ""
                    job.error = f"Part of the audio could not be transcribed: {e}"
                job.done += 1
                realtime.touch()
        job.result = {
            "turns": [t for t in turns if t["text"]],
            "speakers": len({t["speaker"] for t in turns}),
            "seconds": round(seconds, 1),
        }
        job.state = "done"
    except InterruptedError:
        job.state, job.error = "error", "Cancelled."
    except Exception as e:  # noqa: BLE001
        job.state, job.error = "error", str(e) or type(e).__name__


def _start_job(sid, samples):
    if diarizer.state != "ready":
        raise HTTPException(503, diarizer.message or "Speaker separation is not ready yet.")
    if len(samples) < 16000 * 0.5:
        raise HTTPException(400, "The audio is too short.")
    for old in [k for k, j in jobs.items() if j.state != "running"][:-10]:  # keep the last few results
        jobs.pop(old, None)
    job = Job(sid)
    jobs[job.id] = job
    asyncio.create_task(_speaker_job(job, samples))
    return job


@app.post("/api/sessions/{sid}/recordings/{rid}/speakers")
async def recording_speakers(sid: str, rid: str):
    _, rec = _recording_or_404(sid, rid)
    samples = await asyncio.to_thread(_read_wav, rec.path)
    return {"job": _start_job(sid, samples).id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job.")
    return job.to_dict()


def _read_wav(path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


# ---- audio file ---------------------------------------------------------------------
@app.post("/transcribe")
async def transcribe_file(file: UploadFile = File(...), speakers: int = 0, sid: str = ""):
    if status.state not in USABLE_STATES:
        raise HTTPException(503, "The model is not ready yet.")
    suffix = Path(file.filename or "").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    try:
        samples = await asyncio.to_thread(decode_audio, path)
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "Could not read this audio file. Try WAV, MP3, FLAC or OGG.")
    finally:
        os.unlink(path)
    if speakers:
        return {"job": _start_job(sid if valid_sid(sid) else None, samples).id}
    if not await ensure_ready():
        raise HTTPException(503, status.error or "The model is not ready yet.")
    try:
        text = await asyncio.to_thread(realtime.transcribe, samples)
    except BusyError as e:
        raise HTTPException(409, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"Transcription failed: {e}")
    return {"text": text, "seconds": round(len(samples) / 16000, 1)}


# ---- UI -----------------------------------------------------------------------------
@app.get("/favicon.svg")
def favicon_svg():
    return FileResponse(config.STATIC_DIR / "favicon.svg", media_type="image/svg+xml")


@app.get("/favicon.png")
@app.get("/favicon.ico")
def favicon_png():
    return FileResponse(config.STATIC_DIR / "favicon.png", media_type="image/png")


@app.get("/")
def index():
    return FileResponse(config.STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})
