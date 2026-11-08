"""Speaker separation (optional). Runs on the CPU with sherpa-onnx, so the GPU stays free for the ASR model.

Needs `pip install sherpa-onnx` and two small models (about 33 MB) that are downloaded on first start.
If either is missing the rest of the app keeps working; only the "Separate speakers" option is disabled.
"""
import os
import tarfile
import threading

import numpy as np

from . import config
from .models import ModelFile, _download_one

SAMPLE_RATE = 16000
MERGE_GAP = 1.0     # same speaker, pause shorter than this (seconds): one turn
MIN_TURN = 0.5      # turns shorter than this are dropped (too short to transcribe)
PAD_BEFORE = 0.15   # extra real audio around a turn so words at the edges are not cut
PAD_AFTER = 0.25


class Diarizer:
    def __init__(self):
        self.state = "idle"  # idle | preparing | ready | unavailable
        self.message = ""
        self._engine = None
        self._lock = threading.Lock()
        self._file = None  # the download in progress, for the progress text

    def to_dict(self):
        msg = self.message
        f = self._file
        if self.state == "preparing" and f is not None and f.total:
            msg = f"Downloading the speaker models (first time only)... {f.label}: {f.downloaded // 2**20} of {f.total // 2**20} MB"
        return {"state": self.state, "message": msg}

    def _fetch(self, rel_name: str, label: str, url: str, stop):
        """Download one file (or use a copy the user placed in models/diarization under its original name)."""
        dest = config.MODELS_DIR / rel_name
        manual = config.DIAR_DIR / os.path.basename(url)
        if not dest.exists() and manual.exists():
            manual.replace(dest)
            return dest
        if dest.exists():
            return dest
        self._file = ModelFile(rel_name, label, url=url)
        try:
            print(f"Downloading {label} from {url}", flush=True)
            _download_one(self._file, stop, attempts=3)
        finally:
            self._file = None
        return dest

    def _manual_help(self, error) -> str:
        return (
            f"Could not download the speaker models ({error}). Download these two files in a browser, "
            f"put them in the folder {config.DIAR_DIR} (keep their names) and press Retry:\n"
            f"{config.DIAR_SEG_URL}\n{config.DIAR_EMB_URL}"
        )

    # ---- setup (blocking, runs in a background thread) ----------------
    def prepare(self, stop: threading.Event):
        self.state, self.message = "preparing", "Checking the speaker models..."
        try:
            try:
                import sherpa_onnx  # noqa: F401
            except ImportError:
                self.state = "unavailable"
                self.message = "Speaker separation needs: pip install sherpa-onnx (then restart the app)."
                return
            seg_model = config.DIAR_DIR / "segmentation-pyannote-3-0.onnx"
            emb_model = config.DIAR_DIR / "embedding.onnx"
            config.DIAR_DIR.mkdir(parents=True, exist_ok=True)

            if not seg_model.exists():
                self.message = "Downloading the speaker models (first time only)..."
                tar_path = self._fetch("diarization/segmentation.tar.bz2", "segmentation", config.DIAR_SEG_URL, stop)
                with tarfile.open(tar_path, "r:bz2") as tar:
                    member = next((m for m in tar.getmembers() if m.name.endswith("/model.onnx") or m.name == "model.onnx"), None)
                    if member is None:
                        raise RuntimeError("model.onnx not found in the segmentation archive")
                    with tar.extractfile(member) as src, open(seg_model, "wb") as dst:
                        dst.write(src.read())
                tar_path.unlink(missing_ok=True)
            if not emb_model.exists():
                self.message = "Downloading the speaker models (first time only)..."
                self._fetch("diarization/embedding.onnx", "embedding", config.DIAR_EMB_URL, stop)

            self._engine = self._build(seg_model, emb_model)
            self.state, self.message = "ready", ""
        except InterruptedError:
            self.state, self.message = "idle", ""
        except Exception as e:  # noqa: BLE001
            self.state = "unavailable"
            self.message = self._manual_help(e)
            print(self.message, flush=True)

    @staticmethod
    def _build(seg_model, emb_model):
        import os

        import sherpa_onnx as so

        cfg = so.OfflineSpeakerDiarizationConfig(
            segmentation=so.OfflineSpeakerSegmentationModelConfig(
                pyannote=so.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(seg_model)),
                num_threads=min(4, os.cpu_count() or 2),
            ),
            embedding=so.SpeakerEmbeddingExtractorConfig(model=str(emb_model), num_threads=min(4, os.cpu_count() or 2)),
            clustering=so.FastClusteringConfig(num_clusters=-1, threshold=config.DIAR_THRESHOLD),
            min_duration_on=0.3,
            min_duration_off=0.5,
        )
        if not cfg.validate():
            raise RuntimeError("the speaker models could not be loaded")
        return so.OfflineSpeakerDiarization(cfg)

    # ---- work ----------------------------------------------------------
    def run(self, samples: np.ndarray, progress=None):
        """16 kHz mono float32 -> [(start_sec, end_sec, speaker_id)] sorted by start time."""
        if self._engine is None:
            raise RuntimeError(self.message or "Speaker separation is not ready.")

        def cb(done, total):
            # progress() returns True to cancel
            return 1 if (progress and progress(done, total)) else 0

        with self._lock:  # the engine is not re-entrant
            result = self._engine.process(samples, callback=cb).sort_by_start_time()
        return [(float(s.start), float(s.end), int(s.speaker)) for s in result]


def make_turns(segments, total_seconds: float):
    """Raw segments -> turns [{speaker (1-based, by first appearance), start, end}].

    Same-speaker segments separated by a short pause are merged; tiny pieces are dropped.
    """
    order: dict[int, int] = {}
    turns = []
    for start, end, spk in sorted(segments):
        n = order.setdefault(spk, len(order) + 1)
        if turns and turns[-1]["speaker"] == n and start - turns[-1]["end"] <= MERGE_GAP:
            turns[-1]["end"] = max(turns[-1]["end"], end)
        else:
            turns.append({"speaker": n, "start": start, "end": end})
    turns = [t for t in turns if t["end"] - t["start"] >= MIN_TURN]
    # dropping tiny pieces can leave neighbours of the same speaker next to each other
    merged = []
    for t in turns:
        if merged and merged[-1]["speaker"] == t["speaker"] and t["start"] - merged[-1]["end"] <= MERGE_GAP:
            merged[-1]["end"] = t["end"]
        else:
            merged.append(dict(t))
    for t in merged:
        t["start"] = round(max(0.0, t["start"]), 2)
        t["end"] = round(min(total_seconds, t["end"]), 2)
    return merged


def slice_for(samples: np.ndarray, turn) -> np.ndarray:
    """The audio of one turn with a little real audio around it."""
    a = max(0, int((turn["start"] - PAD_BEFORE) * SAMPLE_RATE))
    b = min(len(samples), int((turn["end"] + PAD_AFTER) * SAMPLE_RATE))
    return samples[a:b]
