# stt_service.py – jetzt mit faster-whisper + Logging
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
import numpy as np
from faster_whisper import WhisperModel
import uvicorn
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("stt_service.log"),
        logging.StreamHandler()
    ]
)

app = FastAPI()

# Modellwahl: "base", "small", "medium", "large-v2"
model_size = "small"
device = "cuda" if WhisperModel.is_cuda_available() else "cpu"
logging.info("Loading WhisperModel: %s on %s", model_size, device)
model = WhisperModel(model_size, device=device, compute_type="int8")

@app.post("/transcribe")
async def transcribe(audio: bytes = Body(..., media_type="application/octet-stream")):
    # PCM-Bytes → float32
    audio_np = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / np.iinfo(np.int16).max

    try:
        segments, info = model.transcribe(audio_np, beam_size=5)
        text = " ".join(segment.text.strip() for segment in segments)
        lang = info.language if hasattr(info, "language") else "en"
        logging.info("STT success: lang=%s text='%s'", lang, text.strip())
        return JSONResponse({"language": lang, "text": text.strip()})
    except Exception as e:
        logging.error("STT error: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("stt_service_faster:app", host="0.0.0.0", port=8001, log_level="info")
