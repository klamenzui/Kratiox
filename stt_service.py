# stt_service.py mit faster-whisper
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
import numpy as np
from faster_whisper import WhisperModel
import uvicorn

app = FastAPI()

# Modell laden ("base", "small", "medium", "large-v2")
model_size = "small"
model = WhisperModel(model_size, device="auto", compute_type="int8_float16")

@app.post("/transcribe")
async def transcribe(audio: bytes = Body(..., media_type="application/octet-stream")):
    try:
        # PCM-Bytes → float32 numpy
        audio_np = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / np.iinfo(np.int16).max

        # Transkription durchführen
        segments, info = model.transcribe(audio_np, beam_size=5)

        text = "".join([seg.text for seg in segments])
        return JSONResponse({"language": info.language, "text": text.strip()})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("stt_service:app", host="0.0.0.0", port=8001)
