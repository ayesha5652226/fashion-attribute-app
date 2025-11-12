# app.py
import os
import pickle
from pathlib import Path
from typing import List, Dict

from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import h5py

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = Path("model/finetuned_resnet50_multilabel.h5")
MLB_PATH = Path("model/mlb_dict.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    def __init__(self, backbone, attr_sizes, in_features):
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleList([nn.Linear(in_features, n) for n in attr_sizes])

    def forward(self, x):
        feats = self.backbone(x)
        outs = [torch.sigmoid(h(feats)) for h in self.heads]
        return outs

# -----------------------------
# LOAD MODEL & MLB
# -----------------------------
def load_model_and_mlb(model_path: Path, mlb_path: Path, device):
    with open(mlb_path, "rb") as f:
        mlb_dict = pickle.load(f)

    target_cols = list(mlb_dict.keys())
    attr_sizes = [len(mlb_dict[t].classes_) for t in target_cols]

    backbone = models.resnet50(weights=None)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Identity()
    model = MultiHeadResNet(backbone, attr_sizes, in_features).to(device)

    # Load .h5 weights
    state_dict = {}
    with h5py.File(model_path, "r") as f:
        for ds in f["state_dict"]:
            arr = f["state_dict"][ds][()]
            state_dict[ds.replace("__", "/")] = torch.tensor(arr).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, mlb_dict, target_cols

model, mlb_dict, target_cols = load_model_and_mlb(MODEL_PATH, MLB_PATH, device)

# -----------------------------
# TRANSFORMS
# -----------------------------
VAL_TFMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# -----------------------------
# INFERENCE
# -----------------------------
def predict_image(image: Image.Image):
    x = VAL_TFMS(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outs = model(x)
    results = {}
    for i, col in enumerate(target_cols):
        out = outs[i].cpu().numpy()[0]
        binary = (out >= THRESHOLD).astype(int)
        mlb = mlb_dict[col]
        if binary.sum() > 0:
            labels = mlb.inverse_transform(binary.reshape(1, -1))[0]
        else:
            labels = [mlb.classes_[int(out.argmax())]]
        results[col] = labels
    return results

# -----------------------------
# ROUTES
# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "predictions": None})

@app.post("/", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        return templates.TemplateResponse("index.html", {"request": request, "error": "Invalid file type.", "predictions": None})

    # Save uploaded file temporarily
    temp_path = Path("static/uploads") / file.filename
    os.makedirs(temp_path.parent, exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # Predict
    image = Image.open(temp_path).convert("RGB")
    predictions = predict_image(image)

    # Convert to DataFrame for table rendering
    df = pd.DataFrame(list(predictions.items()), columns=["Attribute", "Predicted Value"])
    table_html = df.to_html(index=False)

    return templates.TemplateResponse("index.html", {"request": request, "predictions": table_html})

# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
