#!/usr/bin/env python3
"""
Kaggle Training — Crack Type Classifier (4-class ResNet-18)
Dataset: RDD 2022 (aliabdelmenam/rdd-2022)
Classes: D10=alligator(0), D20=longitudinal(1), D30=inverse(2), D40=transverse(3)
Runtime: GPU P100/T4 | ~2-3 hours | Output: crack_type_resnet18.pth

SETUP:
1. Add dataset: aliabdelmenam/rdd-2022
2. Enable GPU accelerator (T4 or P100)
"""

# ─── Cell 1: Install ──────────────────────────────────────────────────────────
import os, sys, subprocess

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"]   = "1"

print("Reinstalling PyTorch with cu118 wheel...")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "torch==2.3.1", "torchvision==0.18.1",
    "--index-url", "https://download.pytorch.org/whl/cu118",
    "--upgrade",
], check=False)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm"], check=False)
print("✅ Installations complete.")

# ─── Cell 2: Imports ──────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
from pathlib import Path
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET

print(f"PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    try:
        _t = torch.zeros(1, device="cuda") + 1.0
        assert _t.item() == 1.0
        del _t
        torch.cuda.empty_cache()
        print(f"✅ GPU ready: {torch.cuda.get_device_name(0)}")
        DEVICE = torch.device("cuda")
    except Exception as e:
        print(f"⚠️ Minimal test failed: {e}. Forcing GPU.")
        DEVICE = torch.device("cuda")
else:
    print("No CUDA GPU detected — using CPU")
    DEVICE = torch.device("cpu")

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# ─── Cell 3: Universal Dataset Parser & Crop Extractor ────────────────────────
# Using the base input directory so it finds the images no matter what 
# the dataset folder is named (e.g., 'rdd-2022' vs 'RDD 2022').
RDD_ROOT = Path("/kaggle/input")
CROP_DIR = Path("/kaggle/working/crack_crops")

CLASS_MAP_XML = {
    "D10": "alligator",
    "D20": "longitudinal",
    "D30": "inverse",
    "D40": "transverse",
}

CLASS_MAP_YOLO = {
    0: "alligator",
    1: "longitudinal",
    2: "inverse",
    3: "transverse",
}

for cls_name in CLASS_MAP_XML.values():
    (CROP_DIR / cls_name).mkdir(parents=True, exist_ok=True)

print("Scanning dataset directories for images and labels...")
img_dict = {f.stem: f for f in RDD_ROOT.rglob("*.jpg")}
if not img_dict:
    img_dict = {f.stem: f for f in RDD_ROOT.rglob("*.png")}

xml_dict = {f.stem: f for f in RDD_ROOT.rglob("*.xml")}
txt_dict = {f.stem: f for f in RDD_ROOT.rglob("*.txt") if f.parent.name in ["labels", "train", "val", "test"]}

print(f"Found {len(img_dict)} images, {len(xml_dict)} XML labels, {len(txt_dict)} YOLO TXT labels.")

count = 0
skipped = 0

common_stems = set(img_dict.keys()).intersection(set(xml_dict.keys()).union(txt_dict.keys()))
print(f"Extracting crops from {len(common_stems)} matched image/label pairs...")

for stem in tqdm(list(common_stems), desc="Extracting Crops"):
    img_path = img_dict[stem]
    
    try:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        
        # Priority to XML (Pascal VOC)
        if stem in xml_dict:
            tree = ET.parse(xml_dict[stem])
            root = tree.getroot()
            for obj in root.findall("object"):
                name = obj.find("name").text
                if name in CLASS_MAP_XML:
                    cls_name = CLASS_MAP_XML[name]
                    bndbox = obj.find("bndbox")
                    x1 = int(float(bndbox.find("xmin").text))
                    y1 = int(float(bndbox.find("ymin").text))
                    x2 = int(float(bndbox.find("xmax").text))
                    y2 = int(float(bndbox.find("ymax").text))
                    
                    if x2 <= x1 or y2 <= y1 or (x2-x1)<10 or (y2-y1)<10:
                        continue
                    
                    pad_x, pad_y = int((x2 - x1) * 0.15), int((y2 - y1) * 0.15)
                    crop = img.crop((max(0, x1-pad_x), max(0, y1-pad_y), min(w, x2+pad_x), min(h, y2+pad_y)))
                    crop.save(CROP_DIR / cls_name / f"{stem}_{x1}_{y1}.jpg", quality=90)
                    count += 1
                    
        # Fallback to TXT (YOLO)
        elif stem in txt_dict:
            lines = txt_dict[stem].read_text().strip().split("\n")
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    if cls_id in CLASS_MAP_YOLO:
                        cls_name = CLASS_MAP_YOLO[cls_id]
                        cx, cy, bw, bh = map(float, parts[1:5])
                        x1 = int((cx - bw/2) * w)
                        y1 = int((cy - bh/2) * h)
                        x2 = int((cx + bw/2) * w)
                        y2 = int((cy + bh/2) * h)
                        
                        if x2 <= x1 or y2 <= y1 or (x2-x1)<10 or (y2-y1)<10:
                            continue
                        
                        pad_x, pad_y = int((x2 - x1) * 0.15), int((y2 - y1) * 0.15)
                        crop = img.crop((max(0, x1-pad_x), max(0, y1-pad_y), min(w, x2+pad_x), min(h, y2+pad_y)))
                        crop.save(CROP_DIR / cls_name / f"{stem}_{x1}_{y1}.jpg", quality=90)
                        count += 1
    except Exception as e:
        skipped += 1

print(f"\n✅ Total crops extracted: {count} (Skipped corrupt/empty images: {skipped})")

print("\nCrop distribution per class:")
for cls_name in ["alligator", "longitudinal", "inverse", "transverse"]:
    n = len(list((CROP_DIR / cls_name).glob("*.jpg")))
    bar = "█" * min(n // 500, 30)
    print(f"  {cls_name:15s}: {n:5d}  {bar}")

# ─── Cell 4: Transforms & Dataset ─────────────────────────────────────────────
IMGSZ = 224
BATCH = 32
EPOCHS = 40

from torchvision.datasets import ImageFolder

if count < 10:
    raise RuntimeError("CRITICAL ERROR: Almost no crops were extracted! Check dataset format.")

train_tfm = T.Compose([
    T.RandomResizedCrop(IMGSZ, scale=(0.65, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    T.RandomRotation(30),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_tfm = T.Compose([
    T.Resize((IMGSZ, IMGSZ)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

full_ds = ImageFolder(str(CROP_DIR), transform=train_tfm)
print(f"Classes found: {full_ds.class_to_idx}")

val_sz   = int(0.15 * len(full_ds))
train_sz = len(full_ds) - val_sz
train_ds, val_ds = random_split(full_ds, [train_sz, val_sz])
val_ds.dataset.transform = val_tfm

train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0, pin_memory=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)
print(f"Train samples: {train_sz} | Val samples: {val_sz}")

# ─── Cell 5: Model & Config ───────────────────────────────────────────────────
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 4)
model = model.to(DEVICE)

# Compute class weights for imbalance
counts = {i: 0 for i in range(4)}
for _, lbl in train_ds:
    counts[lbl] += 1
total_train = sum(counts.values())
weights = torch.tensor([total_train / (4 * counts[i] + 1e-8) for i in range(4)]).to(DEVICE)
print(f"\nClass counts: {counts}")
print(f"Class weights: {weights.tolist()}")

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# ─── Cell 6: Training Loop ────────────────────────────────────────────────────
best_acc = 0.0
print("\n===========================================")
print("  Training Crack Type Classifier (4-class)")
print("===========================================\n")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for imgs, labels in tqdm(train_dl, desc=f"Epoch {epoch+1:02d}/{EPOCHS}", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    avg_loss = total_loss / len(train_dl)

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in val_dl:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_acc = correct / max(total, 1)

    mark = ""
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_acc": val_acc,
            "class_map": {0: "alligator", 1: "longitudinal", 2: "inverse", 3: "transverse"},
            "class_to_idx": full_ds.class_to_idx,
        }, "/kaggle/working/crack_type_resnet18.pth")
        mark = "  ✅ Saved"

    print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}{mark}")

print(f"\n{'='*43}")
print(f"  Training complete! Best Acc: {best_acc:.4f}")
print(f"  📥 /kaggle/working/crack_type_resnet18.pth")
print(f"{'='*43}")
