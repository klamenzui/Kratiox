# stt_service.py
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
import whisper
import numpy as np
import uvicorn

app = FastAPI()
model = whisper.load_model("medium")

@app.post("/transcribe")
async def transcribe(audio: bytes = Body(..., media_type="application/octet-stream")):
    # PCM-Bytes → float32 in [-1,1]
    audio_np = np.frombuffer(audio, dtype=np.int16).astype(np.float32)
    audio_np /= np.iinfo(np.int16).max
    result = model.transcribe(audio_np)
    return JSONResponse({"language": result["language"], "text": result["text"].strip()})

if __name__ == "__main__":
    uvicorn.run("stt_service:app", host="0.0.0.0", port=8001)
