"""Bridge between the browser and llama-server's OpenAI-style /v1/realtime WebSocket."""
import asyncio
import base64
import contextlib
import json
import threading
import time

import numpy as np
import websockets
from fastapi import WebSocket

from . import config

SAMPLE_RATE = 16000


class BusyError(RuntimeError):
    pass


def _log(*a):
    if config.DEBUG:
        print("[llama-server]", *a, flush=True)


def _b64(pcm_bytes: bytes) -> str:
    return base64.b64encode(pcm_bytes[: len(pcm_bytes) // 2 * 2]).decode("ascii")


def classify(ev: dict):
    """Turn a llama-server event into (kind, value). kind: text | done | error | None."""
    t = ev.get("type", "")
    if t == "error":
        err = ev.get("error")
        msg = err.get("message") if isinstance(err, dict) else (err or json.dumps(ev))
        return "error", str(msg)
    if t.endswith("transcription.delta") and "delta" in ev:
        return "text", ev.get("delta") or ""
    if t.endswith("transcription.completed") or t == "response.done":
        return "done", ev.get("transcript") or ""
    return None, None


class Realtime:
    def __init__(self):
        self.busy = threading.Lock()
        self.last_used = time.monotonic()  # last request; the idle unload counts from here

    def touch(self):
        self.last_used = time.monotonic()

    def _acquire(self) -> bool:
        ok = self.busy.acquire(blocking=False)
        if ok:
            self.touch()
        return ok

    def _release(self):
        self.touch()
        self.busy.release()

    @contextlib.asynccontextmanager
    async def hold(self, wait: float = 10.0):
        """Own the model for a multi-step job. Waits briefly for a session that is just finishing."""
        waited = 0.0
        while not self._acquire():
            if waited >= wait:
                raise BusyError("The model is busy with another session.")
            await asyncio.sleep(0.2)
            waited += 0.2
        try:
            yield
        finally:
            self._release()

    async def _connect(self):
        conn = await websockets.connect(config.ws_url(), max_size=None, open_timeout=10)
        try:  # first event is session.created (it also reports the audio rate)
            ev = json.loads(await asyncio.wait_for(conn.recv(), timeout=5))
            _log("<-", ev)
            rate = (((ev.get("session") or {}).get("audio") or {}).get("input") or {}).get("format", {}).get("rate")
            if rate and rate != SAMPLE_RATE:
                print(f"WARNING: server expects {rate} Hz audio, this app sends {SAMPLE_RATE} Hz.")
        except asyncio.TimeoutError:
            pass
        return conn

    # ---- live microphone ----------------------------------------------
    async def live(self, ws: WebSocket, session=None):
        """Live microphone. If a tab session is given, the audio is also saved as a recording in it."""
        if not self._acquire():
            await ws.send_json({"type": "error", "message": "The model is busy with another session."})
            await ws.close()
            return

        conn = reader = pending = None
        try:
            try:
                conn = await self._connect()
            except Exception as e:  # noqa: BLE001
                await ws.send_json({"type": "error", "message": f"Cannot reach llama-server ({e})."})
                return

            state = {"got_text": False, "last": time.monotonic()}
            finished = asyncio.Event()

            async def llama_to_browser():
                try:
                    async for raw in conn:
                        state["last"] = time.monotonic()
                        ev = json.loads(raw)
                        _log("<-", ev)
                        kind, val = classify(ev)
                        if kind == "text" and val:
                            state["got_text"] = True
                            await ws.send_json({"type": "text", "text": val})
                        elif kind == "error":
                            await ws.send_json({"type": "error", "message": val})
                        elif kind == "done":
                            if val and not state["got_text"]:
                                await ws.send_json({"type": "text", "text": val})
                            finished.set()
                except websockets.ConnectionClosed:
                    pass
                finally:
                    finished.set()

            reader = asyncio.create_task(llama_to_browser())
            pending = session.start_recording() if session else None
            await ws.send_json({"type": "ready"})

            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes"):
                    if pending:
                        pending.write(msg["bytes"])
                    await conn.send(json.dumps({"type": "input_audio_buffer.append", "audio": _b64(msg["bytes"])}))
                elif msg.get("text") == "stop":
                    await conn.send(json.dumps({"type": "input_audio_buffer.commit"}))
                    state["last"] = time.monotonic()
                    while not finished.is_set():
                        try:
                            await asyncio.wait_for(finished.wait(), timeout=2)
                        except asyncio.TimeoutError:
                            if time.monotonic() - state["last"] > config.IDLE_TIMEOUT:
                                break
                    if pending:
                        rec, pending = session.finish_recording(pending), None
                        if rec:
                            await ws.send_json({"type": "recording", "recording": rec.to_dict()})
                            await session.push()
                    await ws.send_json({"type": "done"})
                    break
        except websockets.ConnectionClosed:
            try:
                await ws.send_json({"type": "error", "message": "Lost connection to llama-server."})
            except Exception:  # noqa: BLE001
                pass
        except Exception as e:  # noqa: BLE001
            print("Live session ended:", repr(e))
        finally:
            if pending:  # the tab went away or an error ended the session: nothing worth keeping
                pending.discard()
            if reader:
                reader.cancel()
            if conn:
                await conn.close()
            self._release()
            try:
                await ws.close()
            except Exception:  # noqa: BLE001
                pass

    # ---- audio file (runs in a worker thread) -------------------------
    def transcribe(self, samples: np.ndarray) -> str:
        if not self._acquire():
            raise BusyError("The model is busy with another session.")
        try:
            return asyncio.run(self._transcribe(samples))
        finally:
            self._release()

    async def _transcribe(self, samples: np.ndarray, idle: float = None) -> str:
        idle = config.IDLE_TIMEOUT if idle is None else idle
        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        conn = await self._connect()
        parts: list[str] = []
        final = {"text": ""}
        finished = asyncio.Event()
        failure: list[str] = []

        async def reader():
            try:
                async for raw in conn:
                    ev = json.loads(raw)
                    _log("<-", ev)
                    kind, val = classify(ev)
                    if kind == "text":
                        parts.append(val)
                    elif kind == "error":
                        failure.append(val)
                        return
                    elif kind == "done":
                        final["text"] = val
                        return
            except websockets.ConnectionClosed:
                pass
            finally:
                finished.set()

        task = asyncio.create_task(reader())
        try:
            step = SAMPLE_RATE * 2  # one second of int16 bytes
            for i in range(0, len(pcm), step):
                await conn.send(json.dumps({"type": "input_audio_buffer.append", "audio": _b64(pcm[i : i + step])}))
            await conn.send(json.dumps({"type": "input_audio_buffer.commit"}))
            last = len(parts)
            while not finished.is_set():
                try:
                    await asyncio.wait_for(finished.wait(), timeout=idle)
                except asyncio.TimeoutError:
                    if len(parts) == last:
                        break
                    last = len(parts)
        finally:
            task.cancel()
            await conn.close()
        if failure and not parts:
            raise RuntimeError(failure[0])
        return "".join(parts) or final["text"]
