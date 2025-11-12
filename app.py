# app.py
import os
import json
import pickle
import threading
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import h5py
from sklearn.preprocessing import MultiLabelBinarizer

# -----------------------------
# CONFIG
# -----------------------------
ROOT = Path.cwd()
MODEL_PATH = ROOT / "model" / "finetuned_resnet50_multilabel.h5"
MLB_PKL_PATH = ROOT / "model" / "mlb_dict.pkl"
MLB_JSON_PATH = ROOT / "model" / "mlb_dict.json"
UPLOAD_DIR = ROOT / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.5

# -----------------------------
# FASTAPI INIT
# -----------------------------
app = FastAPI(title="Fashion Attribute Predictor")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# -----------------------------
# MODEL CLASS
# -----------------------------
class MultiHeadResNet(nn.Module):
    def __init__(self, backbone, attr_sizes: List[int], in_features: int):
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleList([nn.Linear(in_features, n) for n in attr_sizes])

    def forward(self, x):
        feats = self.backbone(x)
        return [torch.sigmoid(h(feats)) for h in self.heads]

# -----------------------------
# TRANSFORMS
# -----------------------------
VAL_TFMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------------
# Lazy load placeholders
# -----------------------------
_model_lock = threading.Lock()
_model = None
_mlb_dict: Dict[str, MultiLabelBinarizer] = None
_target_cols: List[str] = None

# -----------------------------
# Safe MLB loader (pickle -> json fallback)
# -----------------------------
def load_mlb_safe():
    """
    Try to load mlb_dict from pickle; if that fails due to NumPy ABI issues,
    fall back to the JSON file that lists classes per attribute.
    Returns dict(attr_name -> MultiLabelBinarizer)
    """
    # Try pickle first
    if MLB_PKL_PATH.exists():
        try:
            with open(MLB_PKL_PATH, "rb") as f:
                mlb_dict = pickle.load(f)
            return mlb_dict
        except Exception as e:
            # continue to JSON fallback
            print("Warning: failed to load mlb pickle:", repr(e))

    # JSON fallback
    if MLB_JSON_PATH.exists():
        with open(MLB_JSON_PATH, "r", encoding="utf8") as jf:
            raw = json.load(jf)
        mlb_out = {}
        for k, classes in raw.items():
            mlb = MultiLabelBinarizer()
            # fit so mlb.classes_ is set correctly
            mlb.fit([classes])
            mlb_out[k] = mlb
        return mlb_out

    raise FileNotFoundError("No mlb pickle or json found at model/")

# -----------------------------
# Lazy model + mlb loader
# -----------------------------
def get_model_and_mlb():
    global _model, _mlb_dict, _target_cols
    if _model is None or _mlb_dict is None:
        with _model_lock:
            if _model is None or _mlb_dict is None:
                # load mlb dict safely
                _mlb_dict = load_mlb_safe()
                _target_cols = list(_mlb_dict.keys())

                # build backbone and model
                backbone = models.resnet50(weights=None)
                in_features = backbone.fc.in_features
                backbone.fc = nn.Identity()
                attr_sizes = [len(_mlb_dict[t].classes_) for t in _target_cols]
                _model = MultiHeadResNet(backbone, attr_sizes, in_features).to(DEVICE)

                # load weights from HDF5 (.h5)
                state_dict = {}
                with h5py.File(MODEL_PATH, "r") as f:
                    if "state_dict" in f:
                        grp = f["state_dict"]
                    else:
                        grp = f[list(f.keys())[0]]
                    for ds in grp:
                        arr = grp[ds][()]
                        state_dict[ds.replace("__", "/")] = torch.tensor(arr).to(DEVICE)
                _model.load_state_dict(state_dict)
                _model.eval()
    return _model, _mlb_dict, _target_cols

# -----------------------------
# Prediction utility
# -----------------------------
def predict_image(image: Image.Image):
    model, mlb_dict, target_cols = get_model_and_mlb()
    x = VAL_TFMS(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outs = model(x)
    results = {}
    for out, col in zip(outs, target_cols):
        probs = out.cpu().numpy()[0]
        binary = (probs >= THRESHOLD).astype(int)
        mlb = mlb_dict[col]
        if binary.sum() > 0:
            labels = mlb.inverse_transform(binary.reshape(1, -1))[0]
            labels = list(labels) if isinstance(labels, (list, tuple)) else [labels]
        else:
            labels = [mlb.classes_[int(probs.argmax())]]
        results[col] = labels
    return results

# -----------------------------
# Routes
# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "predictions": None})

@app.post("/", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    # basic validation
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        return templates.TemplateResponse("index.html", {"request": request, "error": "Invalid file type", "predictions": None})

    temp_path = UPLOAD_DIR / file.filename
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # Predict
    try:
        image = Image.open(temp_path).convert("RGB")
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "error": f"Failed to open image: {e}", "predictions": None})

    try:
        predictions = predict_image(image)
    except Exception as e:
        # If loading model fails, return a helpful message (don't crash)
        return templates.TemplateResponse("index.html", {"request": request, "error": f"Prediction failed: {e}", "predictions": None})

    df = pd.DataFrame(list(predictions.items()), columns=["Attribute", "Predicted Value"])
    table_html = df.to_html(index=False, header=True, classes="table", escape=False)

    return templates.TemplateResponse("index.html", {"request": request, "predictions": table_html, "image_path": temp_path.name})

@app.get("/health")
def health():
    return {"status": "ok"}

# -----------------------------
# Run the app (for local dev). Render uses the Start Command.
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
