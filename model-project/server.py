# Copyright 2025 Dragos-Stefan Vacarasu
#
# This file was created as part of a modified version of Multi-HMR by NAVER Corp.
# The entire project is licensed under CC BY-NC-SA 4.0.

import os
import uuid
from pathlib import Path

import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from model_service import infer_from_image, load_model

LOCAL_DEV_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
EXTRA = os.getenv("ALLOWED_ORIGINS", "")
origins = LOCAL_DEV_ORIGINS + [o.strip() for o in EXTRA.split(",") if o]

TMP_DIR = Path("tmp_data")
TMP_DIR.mkdir(exist_ok=True)

MODEL_PATH = "logs/threedpw/train_finetune_tinyvit_5m_5e-6/checkpoints/00000.pt"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(TMP_DIR)), name="static")

model = None
device = None

@app.on_event("startup")
def load_model_on_startup():
    """Load the model into memory once, before handling any requests."""
    global model, device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(MODEL_PATH, device=device)
    model.eval()
    print(f"[startup] Loaded model to {device}")  

@app.post("/infer")
async def infer_upload(
    file: UploadFile = File(...),
    threshold: float = Form(0.3),
    fov: int = Form(60)
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    suffix = Path(file.filename).suffix or ".png"
    in_path = TMP_DIR / f"in_{uuid.uuid4().hex}{suffix}"
    data = await file.read()
    in_path.write_bytes(data)

    img = Image.open(in_path).convert("RGB")
    overlay_fp, glb_fp = map(
        Path,
        infer_from_image(model, img, threshold, fov),
    )

    return {
        "overlay_url": f"/static/{overlay_fp.name}",
        "glb_url": f"/static/{glb_fp.name}",
    }

@app.get("/")
async def root():
    return {"status": "ok"}