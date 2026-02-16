import re
import io
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="PillScan API")

# allow React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path(__file__).parent.parent / "notebooks" / "best_model.pth"
DATA_DIR = Path(__file__).parent.parent / "data.nosync" / "pills_raw" / "ogyeiv2"
TRAIN_DIR = DATA_DIR / "train" / "images"
NUM_CLASSES = 112
IMG_SIZE = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load class names from training images (same logic as PillDataset)
# might need to change
def get_class_name(path):
    name = path.stem
    return re.sub(r'_[su]_\d+$', '', name)

image_paths = list(TRAIN_DIR.glob("*.jpg"))
CLASSES = sorted(set(get_class_name(p) for p in image_paths))
print(f"Loaded {len(CLASSES)} classes")

# (same as validation)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

model = load_model()
print(f"Model loaded from {MODEL_PATH}")

@app.get("/")
def root():
    return {"status": "ok", "classes": len(CLASSES)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    class_idx = predicted.item()
    class_name = CLASSES[class_idx]
    confidence_score = confidence.item()

    return {
        "class_name": class_name,
        "confidence": round(confidence_score * 100, 2),
        "class_index": class_idx
    }

@app.get("/classes")
def get_classes():
    return {"classes": CLASSES}
