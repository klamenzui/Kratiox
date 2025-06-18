# tts_service.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from TTS.api import TTS
import torch, os, uuid, uvicorn

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = {
    "de": "tts_models/de/thorsten/vits",
    "en": "tts_models/en/vctk/vits",
    "ru": "tts_models/ru/v3_1/vits",
    "ua": "tts_models/multilingual/multi-dataset/vits",
}
instances = {}

class Req(BaseModel):
    text: str
    lang: str

@app.post("/synthesize")
async def synth(req: Req):
    if req.lang not in MODELS:
        raise HTTPException(status_code=400, detail="unsupported language")

    # Lade oder re-use das TTS-Objekt
    if req.lang not in instances:
        tts = TTS(model_name=MODELS[req.lang], progress_bar=False)
        tts.to(device)
        instances[req.lang] = tts
    tts = instances[req.lang]

    # Entscheide den Speaker (für Multi-Speaker-Modelle)
    tts_kwargs = {"text": req.text}
    if hasattr(tts, "speakers") and tts.speakers:
        tts_kwargs["speaker"] = tts.speakers[0]
    langs = getattr(tts, "languages", []) or []
    if req.lang in langs:
        tts_kwargs["language"] = req.lang

    # Erzeuge eine temporäre WAV-Datei
    tmpfile = f"/tmp/tts_{uuid.uuid4().hex}.wav"
    os.makedirs(os.path.dirname(tmpfile), exist_ok=True)
    tts.tts_to_file(file_path=tmpfile, **tts_kwargs)

    # Gib sie als StreamingResponse zurück
    def iterfile():
        with open(tmpfile, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                yield chunk
        os.remove(tmpfile)

    return StreamingResponse(iterfile(), media_type="audio/wav")

if __name__ == "__main__":
    uvicorn.run("tts_service:app", host="0.0.0.0", port=8003)
