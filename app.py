
# app.py
import io
import h5py
import pickle
import base64
from pathlib import Path
from typing import List, Dict

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import logging

# PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("fashion_app")

ROOT = Path.cwd()
MODEL_PATH = ROOT / "model" / "finetuned_resnet50_multilabel.h5"
MLB_PATH = ROOT / "model" / "mlb_dict.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Using device: {device}")

app = FastAPI(title="Fashion Attribute Detector")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- Load MultiLabelBinarizers ---
if not MLB_PATH.exists():
    raise FileNotFoundError(f"mlb pickle not found at {MLB_PATH}")
with open(MLB_PATH, "rb") as f:
    mlb_dict = pickle.load(f)

target_cols = list(mlb_dict.keys())
log.info(f"Attributes loaded: {target_cols}")
attr_sizes = [len(mlb_dict[c].classes_) for c in target_cols]

# --- Model class (single-linear heads) ---
class MultiHeadResNet(nn.Module):
    def __init__(self, backbone: nn.Module, attr_sizes: List[int], in_features: int):
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleList([nn.Linear(in_features, s) for s in attr_sizes])

    def forward(self, x):
        feats = self.backbone(x)
        return [head(feats) for head in self.heads]

# --- HDF5 -> state_dict loader ---
def load_model_h5_to_state_dict(h5_path: Path, device="cpu"):
    st = {}
    with h5py.File(str(h5_path), "r") as h5f:
        if "state_dict" in h5f:
            grp = h5f["state_dict"]
        elif "model" in h5f:
            grp = h5f["model"]
        else:
            first = list(h5f.keys())[0]
            grp = h5f[first]
        for ds in grp:
            key = ds.replace("__", "/")
            arr = grp[ds][()]
            arr = np.array(arr)
            st[key] = torch.tensor(arr).to(device)
    return st

# --- Build backbone and model ---
backbone = models.resnet50(weights=None)
in_features = backbone.fc.in_features
backbone.fc = nn.Identity()
model = MultiHeadResNet(backbone, attr_sizes, in_features).to(device)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

log.info("Loading weights from HDF5...")
state_dict = load_model_h5_to_state_dict(MODEL_PATH, device=device)
try:
    model.load_state_dict(state_dict, strict=True)
    log.info("Loaded model weights (strict=True).")
except Exception as e:
    log.warning("Strict load failed: %s", e)
    model.load_state_dict(state_dict, strict=False)
model.eval()

# --- Preprocessing transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Pretty names mapping (adjust to your exact target_cols if needed) ---
pretty_names = {
    "PATTERN": "Pattern",
    "COLOUR": "Color",
    "SLEEVE_LENGTH": "Sleeve Length",
    "SLEEVE_TYPE": "Sleeve Type",
    "DRESS_LENGTH": "Dress Length",
    "SIZE": "Size",
    "NECK_LINE": "Neckline",
    "FABRIC (OPTIONAL)": "Fabric",
    "FABRIC": "Fabric",
    "OCASSION": "Occasion",
    "OCCASION": "Occasion",
}

def pretty_attr_name(raw_name: str) -> str:
    return pretty_names.get(raw_name, raw_name.title().replace("_", " "))

# --- Utilities ---
def image_bytes_to_data_url(img_bytes: bytes, img_format="jpeg") -> str:
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/{img_format};base64,{b64}"

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None, "image_data": None})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outs = model(x)
    preds_idx = [int(o.argmax(dim=1).cpu().item()) for o in outs]

    result_map: Dict[str, str] = {}
    for raw_name, idx in zip(target_cols, preds_idx):
        label = mlb_dict[raw_name].classes_[idx]
        nice = pretty_attr_name(raw_name)
        result_map[nice] = str(label)

    fmt = (file.filename.split(".")[-1].lower() if "." in file.filename else "jpeg")
    if fmt not in ("jpg", "jpeg", "png", "gif", "webp"):
        fmt = "jpeg"
    data_url = image_bytes_to_data_url(image_bytes, img_format=fmt if fmt!="jpg" else "jpeg")

    desired_order = ["Pattern", "Color", "Sleeve Length", "Sleeve Type", "Dress Length",
                     "Size", "Neckline", "Fabric", "Occasion"]
    table_rows = []
    for k in desired_order:
        if k in result_map:
            table_rows.append((k, result_map[k]))
    for k, v in result_map.items():
        if k not in set(desired_order):
            table_rows.append((k, v))

    return templates.TemplateResponse("index.html", {"request": request, "result": table_rows, "image_data": data_url})

# --- Corrected download_pdf route (reads multiple 'row' fields) ---
@app.post("/download_pdf")
async def download_pdf(request: Request):
    """
    Expects form fields:
      - image_data: data URL (base64) of the image
      - row: repeated form field "Feature::Value" for each row
    Returns a PDF file as a streaming response.
    """
    form = await request.form()
    image_data = form.get("image_data")
    # Starlette's FormData supports getlist
    rows_list = form.getlist("row") if hasattr(form, "getlist") else []

    # parse rows_list into (feature, value)
    rows = []
    for line in rows_list:
        if not line:
            continue
        if "::" in line:
            k, v = line.split("::", 1)
            rows.append((k.strip(), v.strip()))
        else:
            rows.append((line.strip(), ""))

    # Create PDF in memory
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, height - 60, "Fashion Attribute Prediction Report")

    # Draw uploaded image if present
    if image_data and image_data.startswith("data:image/"):
        try:
            header, b64 = image_data.split(",", 1)
            img_bytes = base64.b64decode(b64)
            img = ImageReader(io.BytesIO(img_bytes))
            img_w, img_h = img.getSize()
            max_w, max_h = 220, 220
            scale = min(max_w / img_w, max_h / img_h, 1.0)
            draw_w, draw_h = img_w * scale, img_h * scale
            c.drawImage(img, 40, height - 60 - draw_h - 10, width=draw_w, height=draw_h)
        except Exception:
            pass

    # Timestamp
    import datetime
    c.setFont("Helvetica", 9)
    c.drawString(40, height - 60 - 240, f"Generated: {datetime.datetime.now().isoformat(sep=' ', timespec='seconds')}")

    # Draw table header and rows to the right of image
    table_x = 300
    table_y_start = height - 100
    c.setFont("Helvetica-Bold", 11)
    c.drawString(table_x, table_y_start, "Feature")
    c.drawString(table_x + 200, table_y_start, "Prediction")
    c.line(table_x, table_y_start - 4, table_x + 380, table_y_start - 4)

    c.setFont("Helvetica", 10)
    y = table_y_start - 20
    row_h = 18
    for (feature, value) in rows:
        if y < 60:
            c.showPage()
            y = height - 80
            c.setFont("Helvetica-Bold", 11)
            c.drawString(table_x, y, "Feature")
            c.drawString(table_x + 200, y, "Prediction")
            y -= 20
            c.setFont("Helvetica", 10)
        c.drawString(table_x, y, feature)
        c.drawString(table_x + 200, y, str(value))
        y -= row_h

    # Disclaimer bottom
    disclaimer = "Disclaimer: This model is a prototype for demo/educational use. Predictions may be inaccurate."
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(40, 40, disclaimer)

    c.showPage()
    c.save()
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="application/pdf",
                             headers={"Content-Disposition": "attachment; filename=prediction_report.pdf"})

@app.get("/health")
def health():
    return {"status": "ok"}
