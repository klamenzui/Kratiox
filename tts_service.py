# tts_service.py - Fixed version with better error handling
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
import os
import uuid
import uvicorn
import tempfile
import traceback

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"TTS Service starting on device: {device}")

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
    try:
        print(f"TTS Request: text='{req.text}', lang='{req.lang}'")

        if req.lang not in MODELS:
            print(f"Unsupported language: {req.lang}")
            raise HTTPException(status_code=400, detail=f"Unsupported language: {req.lang}")

        # Validate input
        if not req.text.strip():
            raise HTTPException(status_code=400, detail="Empty text")

        # Lade oder re-use das TTS-Objekt
        if req.lang not in instances:
            print(f"Loading TTS model for language: {req.lang}")
            try:
                from TTS.api import TTS
                tts = TTS(model_name=MODELS[req.lang], progress_bar=False)
                tts.to(device)
                instances[req.lang] = tts
                print(f"TTS model loaded successfully for {req.lang}")
            except Exception as e:
                print(f"Failed to load TTS model for {req.lang}: {e}")
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Failed to load TTS model: {str(e)}")

        tts = instances[req.lang]

        # Entscheide den Speaker (für Multi-Speaker-Modelle)
        tts_kwargs = {"text": req.text}

        # Check if model has speakers
        if hasattr(tts, 'speakers') and tts.speakers:
            tts_kwargs["speaker"] = tts.speakers[0]
            print(f"Using speaker: {tts.speakers[0]}")

        # Check if model has languages
        languages = getattr(tts, 'languages', None)
        if languages and req.lang in languages:
            tts_kwargs["language"] = req.lang
            print(f"Using language: {req.lang}")

        # Erzeuge eine temporäre WAV-Datei
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmpfile = tmp.name

        try:
            print(f"Synthesizing with kwargs: {tts_kwargs}")
            tts.tts_to_file(file_path=tmpfile, **tts_kwargs)
            print(f"TTS synthesis completed, file: {tmpfile}")

            # Check if file was created and has content
            if not os.path.exists(tmpfile):
                raise HTTPException(status_code=500, detail="TTS file was not created")

            file_size = os.path.getsize(tmpfile)
            if file_size == 0:
                raise HTTPException(status_code=500, detail="TTS file is empty")

            print(f"TTS file size: {file_size} bytes")

            # Gib sie als StreamingResponse zurück
            def iterfile():
                try:
                    with open(tmpfile, "rb") as f:
                        while True:
                            chunk = f.read(4096)
                            if not chunk:
                                break
                            yield chunk
                finally:
                    # Clean up temp file
                    try:
                        os.unlink(tmpfile)
                    except:
                        pass

            return StreamingResponse(iterfile(), media_type="audio/wav")

        except Exception as e:
            # Clean up temp file on error
            try:
                os.unlink(tmpfile)
            except:
                pass
            raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in TTS service: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health():
    return {"status": "healthy", "device": device}


if __name__ == "__main__":
    uvicorn.run("tts_service:app", host="0.0.0.0", port=8003, log_level="info")