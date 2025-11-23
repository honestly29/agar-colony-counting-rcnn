![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![TorchVision](https://img.shields.io/badge/TorchVision-Object%20Detection-orange)
![Faster R-CNN](https://img.shields.io/badge/Model-Faster%20R--CNN-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

# Agar Faster R-CNN CFU Counting

Tools for training and evaluating a Faster R-CNN detector that counts colony-forming units (CFUs) on agar plates. The project crops large plate images into patches, trains on COCO-style annotations, and aggregates predictions back to the plate level with post-processing tuned for counting accuracy.

## Dataset Overview

This project uses the **AGAR dataset**:

Majchrowska, S., Pawłowski, J., Guła, G., Bonus, T., Hanas, A., Loch, A., Pawlak, A., Roszkowiak, J. and Drulis-Kawa, Z. (2021).  
**AGAR: A microbial colony dataset for deep learning detection.** arXiv:2108.01234.

Dataset website:  
https://agar.neurosys.com

## Key Features

- Patch-based dataset loaders for COCO exports or per-image JSON annotations.
- Faster R-CNN with a ResNet-50 backbone and FPN, fine-tuned for colony detection.
- Plate-level evaluation with grid-searched confidence/NMS thresholds, patch and plate metrics, and bin-wise breakdowns.
- Reproducible experiment folders with configs, checkpoints, metrics, and scatter plots.

## Repository Layout

```
.
├── requirements.txt          # Python dependencies (see note below for CPU installs)
├── data/
│   ├── images/               # Plate images used for training/eval
│   └── annotations.json      # COCO annotation file (colonies as bounding boxes)
├── runs/                     # Auto-created experiment outputs (configs, weights, metrics)
└── src/
    ├── datasets/             # Dataset loaders for COCO and per-image JSON formats
    ├── models/               # Faster R-CNN model definition
    ├── pytorch_utils/        # Training engine utilities (ported from torchvision references)
    ├── training/             # Training + evaluation routines
    ├── transforms.py         # Torchvision v2 transforms shared by datasets
    └── sanity_check.py       # Quick inspection of dataset samples
```

## Prerequisites

- Python 3.10 or newer.
- PyTorch 2.8 (GPU preferred for speed). If you are on CPU-only hardware, install the appropriate CPU wheels instead of the CUDA packages listed in `requirements.txt`.
- `pip` or `uv` for dependency management.

## Environment Setup

```bash
# From the repository root
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt  # replace with the CPU variant of torch/torchvision if needed
```

If you prefer to keep the CUDA toolkits managed by `pip`, ensure your NVIDIA drivers support CUDA 12. For CPU-only environments run:

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt --no-deps
```

## Data Preparation

1. Place plate images in `data/images/`.
2. Export annotations in COCO format to `data/annotations.json` (single class named however you like; labels are treated as class index `1`).

### Alternative Annotation Format

If you have per-image JSON files (with `labels` entries containing `x`, `y`, `width`, `height`), use `AGARDataset` from `src/datasets/agar_dataset.py`. Replace the dataset instantiation in your script accordingly.

## Training

Edit the configuration section at the top of `src/training/train.py` to adjust paths, patch size, overlap stride, batch size, and optimizer hyperparameters. Then launch training from the project root:

```bash
python -m src.training.train
```

This script will:

- Split plates into train/val/test without leakage.
- Crop patches, create loaders, and fine-tune Faster R-CNN.
- Save experiment artifacts under `runs/<exp_name>/`, including `config.json`, `weights/`, evaluation logs, and scatter plots.
- Track the best checkpoint based on plate-level MAE.

## Evaluating a Checkpoint

The training script automatically evaluates on the validation split every epoch and on the held-out test set at the end. To re-run evaluation on a saved model:

```bash
python - <<'PY'
from pathlib import Path
import torch
from src.datasets.agar_coco_dataset import AGARCocoDataset
from src.models.faster_rcnn_resnet50_mobilenet_v2 import get_model
from src.pytorch_utils import utils
from src.training.evaluate_counting import evaluate_counting
from src.transforms import get_transform

IMG_DIR = "data/images"
ANN_FILE = "data/annotations.json"
CKPT = Path("runs/your_run/weights/best.pth")

model = get_model(num_classes=2)
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model.eval()

dataset = AGARCocoDataset(IMG_DIR, ANN_FILE, transforms=get_transform(train=False))
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=utils.collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

evaluate_counting(model, loader, device, save_dir=CKPT.parent.parent / "manual_eval", epoch="manual")
PY
```

Metrics (`metrics.csv`) and best post-processing thresholds (`best_postproc.json`) will appear in the chosen `save_dir`.

## Tips

- Keep `PYTHONPATH` pointed at the repository root (running scripts via `python -m ...` does this automatically).
- Patch size and stride control the balance between context and sample count; overlapping patches (`stride < patch_size`) can improve dense colony counts at the cost of more training steps.
- The evaluation grid search (`evaluate_counting.py`) can be tuned via `score_grid`, `iou_grid`, or `use_soft_nms` if your colony density differs.
