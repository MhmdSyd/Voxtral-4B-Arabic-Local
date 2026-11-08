"""
Arabic speech-to-text, one command:   python app.py

Starts a FastAPI server that
  1. downloads the Voxtral Arabic Q8_0 model on first start,
  2. starts llama-server on your GPU,
  3. serves the web UI (live microphone + audio file upload),
and stops llama-server again when you quit (Ctrl+C).
"""
import threading
import webbrowser

import uvicorn

from voxtral import config
from voxtral.server import app

if __name__ == "__main__":
    if config.OPEN_BROWSER:
        threading.Timer(2.0, lambda: webbrowser.open(f"http://localhost:{config.APP_PORT}")).start()
    uvicorn.run(app, host=config.APP_HOST, port=config.APP_PORT)
