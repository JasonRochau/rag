"""
Local Whisper STT server for the Ollama chat UI.

The browser records your headset mic (Poly Blackwire 3320) as webm/opus and
POSTs it here; faster-whisper transcribes it on the GPU and returns text.
Nothing leaves the machine.

Install (you likely already have faster-whisper from the Even glasses build):
    pip install faster-whisper fastapi "uvicorn[standard]" python-multipart

Run:
    python whisper_server.py
Listens on http://localhost:11435  (matches the 'stt' field in the UI).

Env overrides:
    WHISPER_MODEL    default 'small.en'   ('base.en' = faster, 'medium.en' = more accurate)
    WHISPER_DEVICE   default 'cuda'       ('cpu' if VRAM is tight with qwen3:14b loaded)
    WHISPER_COMPUTE  default 'float16'    ('int8_float16' or 'int8' to shrink VRAM use)
    WHISPER_PORT     default '11435'
"""
import os
import tempfile

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import uvicorn

MODEL   = os.environ.get("WHISPER_MODEL", "small.en")
# Default to CPU: sharing GPU VRAM with a loaded Ollama model is what wedges the
# second transcribe. small.en on CPU int8 is plenty fast for short dictation.
# Override with WHISPER_DEVICE=cuda if you want GPU and have the headroom.
DEVICE  = os.environ.get("WHISPER_DEVICE", "cpu")
COMPUTE = os.environ.get("WHISPER_COMPUTE", "int8" if DEVICE == "cpu" else "float16")
PORT    = int(os.environ.get("WHISPER_PORT", "11435"))

print(f"[whisper] loading {MODEL} on {DEVICE} ({COMPUTE}) ...")
try:
    model = WhisperModel(MODEL, device=DEVICE, compute_type=COMPUTE)
except Exception as e:
    print(f"[whisper] GPU load failed ({e}); falling back to CPU int8")
    model = WhisperModel(MODEL, device="cpu", compute_type="int8")
print("[whisper] ready.")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # local tool; the page runs from http://localhost:8000
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL, "device": DEVICE}


@app.post("/transcribe")
def transcribe(file: UploadFile = File(...)):
    # Deliberately a sync 'def' (not 'async def'): model.transcribe() is a
    # blocking call, and FastAPI runs sync endpoints in a threadpool. As an
    # 'async def' the blocking call ran on the event loop and could wedge the
    # server after one request, requiring a restart.
    data = file.file.read()
    print(f"[whisper] /transcribe received {len(data)} bytes", flush=True)
    suffix = os.path.splitext(file.filename or "")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
        path = f.name
    try:
        segments, info = model.transcribe(
            path,
            beam_size=5,
            vad_filter=True,                                  # drop leading/trailing silence
            vad_parameters=dict(min_silence_duration_ms=300),
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        print(f"[whisper] -> {len(text)} chars: {text[:60]!r}", flush=True)
        return {"text": text, "language": info.language}
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT)