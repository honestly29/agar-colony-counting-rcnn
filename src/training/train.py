import torch
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import json
import random

from src.datasets.agar_coco_dataset import AGARCocoDataset
from src.models.faster_rcnn_resnet50_mobilenet_v2 import get_model
from src.pytorch_utils.engine import train_one_epoch
from src.pytorch_utils import utils
from src.training.evaluate_counting import evaluate_counting
from src.transforms import get_transform


# === CONFIG ===
IMG_DIR = "data/images"
ANN_FILE = "data/annotations.json"

PATCH_SIZE = 512     # crop size for patches
STRIDE = 512         # step size between crops (can be < PATCH_SIZE for overlap)
INCLUDE_EMPTY = True # include empty patches in training

BATCH_SIZE = 2
NUM_WORKERS = 2
NUM_EPOCHS = 3
LR = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
STEP_SIZE = 3
GAMMA = 0.1


# === Device ===
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
else:
    device = torch.device("cpu")
    print("CUDA not available. Using CPU.")


# === Dataset ===
dataset = AGARCocoDataset(
    IMG_DIR, ANN_FILE, transforms=get_transform(train=True),
    patch_size=PATCH_SIZE, stride=STRIDE, include_empty=INCLUDE_EMPTY
)
dataset_eval = AGARCocoDataset(
    IMG_DIR, ANN_FILE, transforms=get_transform(train=False),
    patch_size=PATCH_SIZE, stride=STRIDE, include_empty=True
)

# === Plate-level split (no leakage) ===
all_plate_ids = dataset.ids  # COCO image_ids
random.shuffle(all_plate_ids)

n_total = len(all_plate_ids)
n_train = int(0.75 * n_total)
n_val = int(0.15 * n_total)
n_test = n_total - n_train - n_val

train_plate_ids = set(all_plate_ids[:n_train])
val_plate_ids   = set(all_plate_ids[n_train:n_train+n_val])
test_plate_ids  = set(all_plate_ids[n_train+n_val:])

def expand_to_patch_indices(ds, plate_ids):
    """Expand plate IDs into patch indices for Subset."""
    return [
        i for i, (img_idx, *_)
        in enumerate(ds.index_map)
        if ds.ids[img_idx] in plate_ids
    ]

train_indices = expand_to_patch_indices(dataset, train_plate_ids)
val_indices   = expand_to_patch_indices(dataset_eval, val_plate_ids)
test_indices  = expand_to_patch_indices(dataset_eval, test_plate_ids)

dataset_train = Subset(dataset, train_indices)
dataset_val   = Subset(dataset_eval, val_indices)
dataset_test  = Subset(dataset_eval, test_indices)

print(f"Plates: train={len(train_plate_ids)}, val={len(val_plate_ids)}, test={len(test_plate_ids)}")
print(f"Patches: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")


# === DataLoaders ===
data_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True,
                         collate_fn=utils.collate_fn, num_workers=NUM_WORKERS)
data_loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False,
                             collate_fn=utils.collate_fn)
data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False,
                              collate_fn=utils.collate_fn)


# === Model ===
model = get_model(num_classes=2)  # background + colony
model.to(device)


# === Optimizer & Scheduler ===
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)


# === Output directory (runs/...) ===
runs_dir = Path("runs")
runs_dir.mkdir(exist_ok=True)

exp_name = f"patch{PATCH_SIZE}_stride{STRIDE}_bs{BATCH_SIZE}_lr{LR}_epochs{NUM_EPOCHS}"
existing = [p for p in runs_dir.glob(f"{exp_name}*") if p.is_dir()]
exp_id = len(existing) + 1
exp_dir = runs_dir / f"{exp_name}_exp{exp_id}"
exp_dir.mkdir(parents=True, exist_ok=True)
(exp_dir / "weights").mkdir(parents=True, exist_ok=True)
print(f"Saving outputs to {exp_dir}")

# Save config
config = {
    "IMG_DIR": IMG_DIR,
    "ANN_FILE": ANN_FILE,
    "PATCH_SIZE": PATCH_SIZE,
    "STRIDE": STRIDE,
    "INCLUDE_EMPTY": INCLUDE_EMPTY,
    "BATCH_SIZE": BATCH_SIZE,
    "NUM_WORKERS": NUM_WORKERS,
    "NUM_EPOCHS": NUM_EPOCHS,
    "LR": LR,
    "MOMENTUM": MOMENTUM,
    "WEIGHT_DECAY": WEIGHT_DECAY,
    "STEP_SIZE": STEP_SIZE,
    "GAMMA": GAMMA,
    "device": str(device),
    "train_plates": len(train_plate_ids),
    "val_plates": len(val_plate_ids),
    "test_plates": len(test_plate_ids),
}
with open(exp_dir / "config.json", "w") as f:
    json.dump(config, f, indent=4)


# === Training loop ===
best_mae = float("inf")

for epoch in range(NUM_EPOCHS):
    # Train
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    lr_scheduler.step()

    # Evaluate on validation set
    _, (mae_plate, rmse_plate, bias_plate, msle_plate, smape_plate), _ = evaluate_counting(
        model, data_loader_val, device,
        save_dir=exp_dir / "evaluate_counting_results", epoch=epoch + 1
    )

    # Save checkpoint (last)
    ckpt_path = exp_dir / "weights" / "last.pth"
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
    }, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

    # Save best (based on plate-level MAE)
    if mae_plate < best_mae:
        best_mae = mae_plate
        best_path = exp_dir / "weights" / "best.pth"
        torch.save(model.state_dict(), best_path)
        print(f"New best model saved (MAE_plate={mae_plate:.3f}) to {best_path}")

# Save final model
final_path = exp_dir / "weights" / "final.pth"
torch.save(model.state_dict(), final_path)
print(f"Training complete. Final model saved as {final_path}")

# === Final evaluation on test set ===
print("\n=== Final Test Evaluation ===")
evaluate_counting(model, data_loader_test, device,
                  save_dir=exp_dir / "evaluate_counting_results", epoch="test")

