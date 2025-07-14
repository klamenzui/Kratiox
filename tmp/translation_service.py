# translation_service.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import MarianTokenizer, MarianMTModel
import torch, uvicorn

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = {
    "de-en": "Helsinki-NLP/opus-mt-de-en",
    "en-de": "Helsinki-NLP/opus-mt-en-de",
    "ru-en": "Helsinki-NLP/opus-mt-ru-en",
    "en-ru": "Helsinki-NLP/opus-mt-en-ru",
    "uk-en": "Helsinki-NLP/opus-mt-mul-en",
    "en-uk": "Helsinki-NLP/opus-mt-en-mul",
}
pipelines = {}

class Req(BaseModel):
    text: str
    src: str
    dest: str

@app.post("/translate")
async def translate(req: Req):
    key = f"{req.src}-{req.dest}"
    if key not in MODELS:
        return {"error": "unsupported language pair"}, 400
    if key not in pipelines:
        tok = MarianTokenizer.from_pretrained(MODELS[key])
        mdl = MarianMTModel.from_pretrained(MODELS[key]).to(device)
        pipelines[key] = (tok, mdl)
    tok, mdl = pipelines[key]
    batch = tok(req.text, return_tensors="pt", padding=True).to(device)
    tokens = mdl.generate(**batch)
    out = tok.batch_decode(tokens, skip_special_tokens=True)[0]
    return {"translation": out}

if __name__ == "__main__":
    uvicorn.run("translation_service:app", host="0.0.0.0", port=8002)
