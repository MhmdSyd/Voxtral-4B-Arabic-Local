# Voxtral Realtime Arabic (local)

Arabic speech-to-text on your own GPU: live microphone and audio-file upload.
One command starts everything.

```
python app.py
```

The app (FastAPI) does the whole job:
1. checks that llama.cpp is installed,
2. downloads the **Q8_0** model and its audio encoder on the first start (about 4.7 GB, resumable),
3. starts `llama-server` on the GPU,
4. opens the web UI at http://localhost:8000,
5. stops `llama-server` when you press Ctrl+C.

## What the app does

- **Live microphone** with a readable transcript: punctuation (. ، ؟ !), sentence breaks and paragraphs.
- **Recordings**: every live recording is saved and can be played again, downloaded or deleted.
- **Tab sessions**: each browser tab is its own session. Its recordings (and transcript) are kept while the tab is open,
  also across a page refresh, and are deleted shortly after the tab is closed.
- **Separate speakers** (optional): the transcript is written as `Speaker 1: ...`, `Speaker 2: ...`, one line per turn.
- **Idle unload**: after 2 minutes without a request the model is removed from the GPU. It loads again by itself
  (a few seconds) when you press Start or upload a file.

## Setup (once)

```powershell
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

winget install llama.cpp          # needs build b11675 / v0.8.0 or newer. Open a new terminal afterwards.
python app.py
```

If `winget` is not an option, download a CUDA build from
https://github.com/ggml-org/llama.cpp/releases and either put it on your PATH,
unpack it into a `bin` folder in this project, or set `LLAMA_SERVER` to the full path of `llama-server.exe`.

If something is missing or fails, the UI shows what went wrong (with the last lines of
`logs/llama-server.log`) and a **Retry** button.

## Project layout

```
app.py                  entry point
voxtral/config.py       all settings
voxtral/models.py       model download with progress and resume
voxtral/llama_manager.py  find / start / watch / stop llama-server
voxtral/realtime.py     browser <-> llama-server realtime bridge
voxtral/sessions.py     tab sessions and saved recordings
voxtral/diarize.py      speaker separation (sherpa-onnx, CPU)
voxtral/server.py       FastAPI app: setup state, API, WebSocket, upload, idle unload
static/index.html       the UI (+ favicon.svg / favicon.png)
sessions/               saved recordings of the open tabs (deleted when a tab closes)
models/                 downloaded GGUF files (created on first start)
logs/llama-server.log   llama-server output
```

## Settings (environment variables, all optional)

| Variable | Default | Meaning |
|---|---|---|
| `PORT` | 8000 | web app port |
| `LLAMA_PORT` | 8080 | llama-server port |
| `LLAMA_SERVER` | (auto) | full path to llama-server |
| `CTX_SIZE` | 16384 | about 12.5 tokens per second of audio, so roughly 20 minutes per live session |
| `MMPROJ_FILE` | `...mmproj-Q8_0.gguf` | use `Voxtral-Mini-4B-Realtime-Arabic.mmproj-f16.gguf` for the full-precision audio encoder |
| `MODELS_DIR` | `./models` | where model files are stored |
| `OPEN_BROWSER` | 1 | set to 0 to not open the browser |
| `IDLE_UNLOAD_SECONDS` | 120 | remove the model from the GPU after this many seconds without a request (0 = never) |
| `SESSION_GRACE_SECONDS` | 30 | recordings are deleted this long after the tab is closed (so a refresh keeps them) |
| `DIAR_THRESHOLD` | 0.5 | speaker separation: lower finds more speakers, higher merges them |
| `DEBUG` | 0 | set to 1 to print every event from llama-server |

If a `llama-server` is already running on `LLAMA_PORT`, the app uses it instead of starting its own.

## Notes
- Use `localhost` (not an IP address): browsers only allow the microphone on localhost or HTTPS.
- Live and file transcription share one model, so one runs at a time.
- File upload accepts WAV, MP3, FLAC and OGG.
- Transcripts are saved only when you press **Save .txt**. Recordings are saved on disk in `sessions/` only while their tab is open.
- Punctuation is added by rules (pauses between words, question words, very long runs), not by the model, so it is a good
  guess and not perfect. Existing punctuation from the model is kept.
- **Separate speakers** needs `sherpa-onnx` (in requirements.txt) and downloads two small models (about 33 MB) on first start.
  Live recordings are separated when you press Stop; each speaker turn is then transcribed again, so the final text can differ
  slightly from what you saw live. Overlapping voices, very short remarks and similar voices can be assigned to the wrong speaker.
  If the models cannot be downloaded, the option is disabled and everything else keeps working.
- The idle unload only applies to the llama-server this app started. A llama-server you started yourself is left alone.
