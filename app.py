
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import shutil
import pickle
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os

# ----------------------------
# Configuration
# ----------------------------
MODEL_PATH = Path("model/finetuned_resnet50_multilabel.h5")
MLB_PATH   = Path("model/mlb_dict.pkl")
UPLOAD_DIR = Path("static/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.5

# ----------------------------
# Load mlb_dict
# ----------------------------
with open(MLB_PATH, "rb") as f:
    mlb_dict = pickle.load(f)
target_cols = list(mlb_dict.keys())

# ----------------------------
# Model class
# ----------------------------
class MultiHeadResNet(nn.Module):
    def __init__(self, backbone, attr_sizes, in_features):
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleList([nn.Linear(in_features, n) for n in attr_sizes])

    def forward(self, x):
        feats = self.backbone(x)
        return [torch.sigmoid(h(feats)) for h in self.heads]

# ----------------------------
# Load model
# ----------------------------
def load_model():
    backbone = models.resnet50(weights=None)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Identity()
    attr_sizes = [len(mlb_dict[c].classes_) for c in target_cols]
    model = MultiHeadResNet(backbone, attr_sizes, in_features).to(DEVICE)

    import h5py
    st = {}
    with h5py.File(MODEL_PATH, "r") as f:
        for ds in f["state_dict"]:
            arr = f["state_dict"][ds][()]
            st[ds.replace("__", "/")] = torch.tensor(arr).to(DEVICE)
    model.load_state_dict(st)
    model.eval()
    return model

model = load_model()

# ----------------------------
# Transform
# ----------------------------
VAL_TFMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "results": None})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    # Save uploaded image
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Predict
    img = Image.open(file_path).convert("RGB")
    x = VAL_TFMS(img).unsqueeze(0).to(DEVICE)
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
            labels = [mlb.classes_[int(np.argmax(probs))]]
        results[col] = labels

    return templates.TemplateResponse("index.html", {"request": request, "results": results, "image_path": file_path.name})
