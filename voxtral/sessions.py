"""Browser-tab sessions.

Every browser tab has its own session id. While the tab is open it keeps a WebSocket to
/ws/session. The recordings made in that tab are saved in sessions/<id>/ and are deleted
SESSION_GRACE seconds after that WebSocket closes (tab closed, browser quit, network lost).
The short delay lets a page refresh keep its recordings.
"""
import asyncio
import re
import shutil
import time
import wave
from dataclasses import dataclass
from pathlib import Path

from . import config

SAMPLE_RATE = 16000
_SID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
_RID = re.compile(r"^r\d{1,6}$")


def valid_sid(sid) -> bool:
    return bool(sid and _SID.match(sid))


def valid_rid(rid) -> bool:
    return bool(rid and _RID.match(rid))


@dataclass
class Recording:
    id: str
    name: str
    path: Path
    seconds: float
    size: int
    created: float  # unix time

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "seconds": round(self.seconds, 1),
            "size": self.size,
            "created": int(self.created),
        }


class PendingRecording:
    """A recording that is being written (16 kHz mono 16-bit WAV, the same audio the model hears)."""

    def __init__(self, rid: str, name: str, path: Path):
        self.id, self.name, self.path = rid, name, path
        self.created = time.time()
        self.frames = 0
        self._w = wave.open(str(path), "wb")
        self._w.setnchannels(1)
        self._w.setsampwidth(2)
        self._w.setframerate(SAMPLE_RATE)

    def write(self, pcm: bytes):
        pcm = pcm[: len(pcm) // 2 * 2]
        if pcm:
            self._w.writeframes(pcm)
            self.frames += len(pcm) // 2

    def close(self):
        try:
            self._w.close()
        except Exception:  # noqa: BLE001
            pass

    def discard(self):
        self.close()
        self.path.unlink(missing_ok=True)


class Session:
    def __init__(self, sid: str):
        self.id = sid
        self.dir = config.SESSIONS_DIR / sid
        self.recordings: dict[str, Recording] = {}
        self.sockets: set = set()
        self.counter = 0
        self.timer = None

    def start_recording(self) -> PendingRecording:
        self.counter += 1
        self.dir.mkdir(parents=True, exist_ok=True)
        rid = f"r{self.counter}"
        return PendingRecording(rid, f"Recording {self.counter}", self.dir / f"{rid}.wav")

    def finish_recording(self, pending: PendingRecording):
        """Close the file and register it. Returns the recording, or None if it was too short."""
        pending.close()
        seconds = pending.frames / SAMPLE_RATE
        if seconds < config.MIN_RECORDING_SECONDS:
            pending.path.unlink(missing_ok=True)
            return None
        rec = Recording(pending.id, pending.name, pending.path, seconds, pending.path.stat().st_size, pending.created)
        self.recordings[rec.id] = rec
        return rec

    def delete_recording(self, rid: str) -> bool:
        rec = self.recordings.pop(rid, None)
        if rec:
            rec.path.unlink(missing_ok=True)
        return rec is not None

    def to_dict(self):
        return {
            "type": "session",
            "id": self.id,
            "recordings": [r.to_dict() for r in self.recordings.values()],
        }

    async def push(self):
        """Send the current recording list to every socket this tab has open."""
        for ws in list(self.sockets):
            try:
                await ws.send_json(self.to_dict())
            except Exception:  # noqa: BLE001
                self.sockets.discard(ws)


class SessionStore:
    def __init__(self):
        self.sessions: dict[str, Session] = {}
        self.on_expire = []  # callbacks(sid)

    def wipe_all(self):
        shutil.rmtree(config.SESSIONS_DIR, ignore_errors=True)
        config.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    def get(self, sid: str) -> Session:
        s = self.sessions.get(sid)
        if s is None:
            s = self.sessions[sid] = Session(sid)
            self._schedule(s)  # expires unless a tab connects
        return s

    def find(self, sid: str):
        return self.sessions.get(sid) if valid_sid(sid) else None

    def attach(self, sid: str, ws) -> Session:
        s = self.get(sid)
        s.sockets.add(ws)
        self._cancel(s)
        return s

    def detach(self, sid: str, ws):
        s = self.sessions.get(sid)
        if s is None:
            return
        s.sockets.discard(ws)
        if not s.sockets:
            self._schedule(s)

    def _cancel(self, s: Session):
        if s.timer:
            s.timer.cancel()
            s.timer = None

    def _schedule(self, s: Session):
        self._cancel(s)
        s.timer = asyncio.get_running_loop().call_later(config.SESSION_GRACE, self.expire, s.id)

    def expire(self, sid: str):
        s = self.sessions.get(sid)
        if s is None or s.sockets:
            return
        self.sessions.pop(sid, None)
        self._cancel(s)
        for cb in self.on_expire:
            try:
                cb(sid)
            except Exception:  # noqa: BLE001
                pass
        shutil.rmtree(s.dir, ignore_errors=True)
