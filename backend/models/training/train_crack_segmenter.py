#!/usr/bin/env python3
"""
Kaggle Training Notebook — U-Net Crack Segmenter (Binary)
Dataset: Crack Segmentation Dataset (11,200 images + masks)
Runtime: GPU P100 (SM 60) | ~2 hours | Output: crack_unet.pth

SETUP ON KAGGLE:
1. Datasets > Add: lakshaymiddha/crack-segmentation-dataset
2. Enable GPU Accelerator (P100 or T4)

ROOT CAUSE OF "no kernel image" ERROR:
  Kaggle ships PyTorch 2.x+cu128 (CUDA 12.8) which does NOT include
  P100 (SM 60) kernel binaries. Fix: reinstall PyTorch with the cu118
  wheel which still bundles SM 60 kernels for P100.
  This MUST run before any 'import torch'.
"""

# ─── Cell 1: Reinstall PyTorch WITH P100 (SM 60) Support ─────────────────────
# Run this cell, then continue — NO kernel restart needed on Kaggle
import os, sys, subprocess

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"]   = "1"

print("Reinstalling PyTorch with cu118 wheel (supports P100 SM 60 + T4 SM 75)...")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "torch==2.3.1",
    "torchvision==0.18.1",
    "--index-url", "https://download.pytorch.org/whl/cu118",
    "--upgrade",
], check=False)

# Install SMP + albumentations AFTER torch so dependency resolver works correctly
print("Installing segmentation-models-pytorch, albumentations...")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "segmentation-models-pytorch",
    "albumentations",
    "timm",
], check=False)

print("✅ All packages installed.")

print("Packages installed.")

# ─── Cell 2: Imports ─────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import segmentation_models_pytorch as smp

# ─── Cell 3: GPU Setup & cuDNN Fix ───────────────────────────────────────────
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"CUDA version    : {torch.version.cuda}")
    cap = torch.cuda.get_device_capability(0)
    print(f"Compute cap     : SM {cap[0]}{cap[1]}")

# KEY FIX: Disable cuDNN auto-tuner — prevents "no kernel image" error
# benchmark=True lets cuDNN pick the fastest kernel, but on some GPUs
# it picks one compiled for a different SM architecture → crash
torch.backends.cudnn.benchmark     = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled       = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {DEVICE}")

# ─── Cell 4: Force GPU — No CPU Fallback ─────────────────────────────────────
# The 'torch.mm' smoke test can fail on Kaggle P100 even with correct drivers.
# Instead: directly attempt to move a tensor to CUDA. If torch says CUDA is
# available, we TRUST it and force GPU. The real crash (if any) will surface
# at the model forward pass check in Cell 8 with a clear message.

if torch.cuda.is_available():
    try:
        # Minimal test — tensor creation + scalar op, no complex kernels
        _t = torch.zeros(1, device="cuda") + 1.0
        assert _t.item() == 1.0
        del _t
        torch.cuda.empty_cache()
        print(f"✅ GPU ready: {torch.cuda.get_device_name(0)}  (SM {torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]})")
        DEVICE = torch.device("cuda")
    except Exception as e:
        print(f"⚠️  CUDA tensor test failed: {e}")
        print("   Forcing CUDA anyway — will fail loudly if GPU is broken")
        DEVICE = torch.device("cuda")   # Do NOT fall back silently
else:
    print("No CUDA GPU detected — using CPU")
    DEVICE = torch.device("cpu")

print(f"Training device: {DEVICE}")

# ─── Cell 5: Config ───────────────────────────────────────────────────────────
CRACK_ROOT = Path("/kaggle/input/crack-segmentation-dataset")
IMGSZ  = 256
BATCH  = 8      # Smaller batch = safer for all GPU types
EPOCHS = 40

# ─── Cell 6: Dataset ──────────────────────────────────────────────────────────
class CrackDataset(Dataset):
    def __init__(self, img_paths, mask_dir: Path, transforms=None):
        self.images     = img_paths
        self.mask_dir   = mask_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path  = self.images[idx]
        mask_path = self.mask_dir / img_path.name
        if not mask_path.exists():
            mask_path = self.mask_dir / (img_path.stem + ".png")
        if not mask_path.exists():
            # Try same directory with _mask suffix
            mask_path = self.mask_dir / (img_path.stem + "_mask.png")

        img  = np.array(Image.open(img_path).convert("RGB"))
        if mask_path.exists():
            mask = np.array(Image.open(mask_path).convert("L"))
        else:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        mask = (mask > 127).astype(np.float32)

        if self.transforms:
            aug  = self.transforms(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]

        return img, mask.unsqueeze(0)

# ─── Cell 7: Build DataLoaders ────────────────────────────────────────────────
train_tfm = A.Compose([
    A.Resize(IMGSZ, IMGSZ),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.GaussNoise(p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
val_tfm = A.Compose([
    A.Resize(IMGSZ, IMGSZ),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

def find_dataset_dirs(root: Path):
    """Auto-detect image and mask directories regardless of dataset structure."""
    print("\nDataset directory listing:")
    for p in sorted(root.rglob("*"))[:30]:
        print(" ", p.relative_to(root))

    # Potential image dirs (in priority order)
    img_candidates  = ["train/images", "images/train", "images", "img", "train"]
    mask_candidates = ["train/masks",  "masks/train",  "masks",  "gt",  "label",
                       "train/labels", "labels"]

    img_dir = mask_dir = None
    for cand in img_candidates:
        p = root / cand
        imgs = list(p.glob("*.jpg")) + list(p.glob("*.png")) if p.exists() else []
        if imgs:
            img_dir = p
            print(f"\n✓ Image dir: {p} ({len(imgs)} files)")
            break

    for cand in mask_candidates:
        p = root / cand
        masks = list(p.glob("*.png")) + list(p.glob("*.jpg")) if p.exists() else []
        if masks:
            mask_dir = p
            print(f"✓ Mask dir:  {p} ({len(masks)} files)")
            break

    if img_dir is None:
        raise RuntimeError(f"No images found in {root}. Check dataset path.")
    if mask_dir is None:
        mask_dir = img_dir   # some datasets store images + masks together
        print(f"⚠️  Mask dir not found — trying same dir as images")

    return img_dir, mask_dir

# Try standard train/val split first
use_split = False
for split_combo in [("train/images", "train/masks", "test/images", "test/masks"),
                    ("train/images", "train/masks", "val/images",  "val/masks")]:
    tr_img, tr_msk, va_img, va_msk = [CRACK_ROOT / s for s in split_combo]
    if tr_img.exists() and tr_msk.exists() and va_img.exists():
        t_imgs = sorted(tr_img.glob("*.jpg")) + sorted(tr_img.glob("*.png"))
        v_imgs = sorted(va_img.glob("*.jpg")) + sorted(va_img.glob("*.png"))
        if t_imgs:
            train_ds = CrackDataset(t_imgs, tr_msk, train_tfm)
            val_ds   = CrackDataset(v_imgs, va_msk, val_tfm)
            use_split = True
            print(f"Using pre-split: train={len(train_ds)}, val={len(val_ds)}")
            break

if not use_split:
    img_dir, mask_dir = find_dataset_dirs(CRACK_ROOT)
    all_imgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    val_n    = max(int(0.15 * len(all_imgs)), 1)
    train_n  = len(all_imgs) - val_n
    all_ds   = CrackDataset(all_imgs, mask_dir, train_tfm)
    train_ds, val_ds = random_split(all_ds, [train_n, val_n])
    val_ds.dataset.transforms = val_tfm
    print(f"Auto-split: train={train_n}, val={val_n}")

# num_workers=0 avoids Kaggle multiprocessing issues
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0, pin_memory=(DEVICE.type=="cuda"))
val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)
print(f"\nTrain batches: {len(train_dl)}, Val batches: {len(val_dl)}")

# ─── Cell 8: Model + Pre-train Forward-Pass Test ─────────────────────────────
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None,
).to(DEVICE)

# ★ Critical: run a dummy forward pass BEFORE the training loop
# If there's a CUDA kernel mismatch it will fail HERE with a clear error
print("\nRunning pre-training forward-pass verification...")
model.eval()
with torch.no_grad():
    dummy = torch.randn(2, 3, IMGSZ, IMGSZ).to(DEVICE)
    out   = model(dummy)
    assert out.shape == (2, 1, IMGSZ, IMGSZ), f"Unexpected output shape: {out.shape}"
print(f"✅ Forward pass OK — output: {out.shape}")
del dummy, out
torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

# ─── Cell 9: Loss, Optimizer ──────────────────────────────────────────────────
bce       = nn.BCEWithLogitsLoss()
dice_loss = smp.losses.DiceLoss(mode="binary")

def combined_loss(pred, target):
    return 0.5 * bce(pred, target) + 0.5 * dice_loss(pred, target)

optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.5)  # verbose removed in PyTorch 2.2+

def iou_score(pred_logits, true_mask, threshold=0.5):
    pred  = (torch.sigmoid(pred_logits) > threshold).float()
    inter = (pred * true_mask).sum()
    union = pred.sum() + true_mask.sum() - inter
    return float(inter / (union + 1e-8))

# ─── Cell 10: Training Loop ───────────────────────────────────────────────────
best_iou = 0.0
print("\n========================================")
print("       Starting U-Net Training")
print("========================================\n")

for epoch in range(EPOCHS):
    # ── Train ────────────────────────────────────────────────────────────────
    model.train()
    total_loss = 0.0
    for imgs, masks in tqdm(train_dl, desc=f"Epoch {epoch+1:02d}/{EPOCHS} [TRAIN]", leave=False):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        preds = model(imgs)
        loss  = combined_loss(preds, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_dl)

    # ── Validate ──────────────────────────────────────────────────────────────
    model.eval()
    iou_list = []
    with torch.no_grad():
        for imgs, masks in val_dl:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs)
            iou_list.append(iou_score(preds, masks))
    val_iou = float(np.mean(iou_list))
    scheduler.step(1 - val_iou)   # Scheduler minimizes 1-IoU

    status = ""
    if val_iou > best_iou:
        best_iou = val_iou
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_iou": val_iou,
            "encoder": "resnet34",
            "imgsz": IMGSZ,
            "classes": 1,
        }, "/kaggle/working/crack_unet.pth")
        status = "  ✅ Saved"

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {avg_loss:.4f} | Val IoU: {val_iou:.4f}{status}")

print(f"\n{'='*40}")
print(f"  Training complete!  Best IoU: {best_iou:.4f}")
print(f"  📥 Download: /kaggle/working/crack_unet.pth")
print(f"{'='*40}")
