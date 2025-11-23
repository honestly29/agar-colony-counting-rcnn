import warnings
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.agar_coco_dataset import AGARCocoDataset


IMG_DIR = Path("data/images")  # update if needed
ANN_FILE = Path("data/annotations.json")


def collate_fn(batch):
    batch = [(img, tgt) for img, tgt in batch if img is not None]
    if len(batch) == 0:
        return [], []
    return tuple(zip(*batch))


def main():
    print("Using annotation file:", ANN_FILE.resolve())
    print("Using image directory:", IMG_DIR.resolve())

    dataset = AGARCocoDataset(IMG_DIR, ANN_FILE)
    print("Dataset length:", len(dataset))

    # Show first 5 expected image paths
    for i in range(5):
        ann_id = dataset.ids[i]
        file_name = dataset.coco.loadImgs(ann_id)[0]["file_name"]
        print(f"Expected path {i}: {IMG_DIR / file_name}")

    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

    for img, target in loader:
        if len(img) == 0:
            warnings.warn("No valid images found yet, check IMG_DIR")
            break

        img = img[0]
        target = target[0]

        if torch.is_tensor(img):
            print("Image tensor shape:", img.shape)
        else:
            print("PIL image size:", img.size)

        print("Target keys:", list(target.keys()))
        print("Boxes tensor:", target["boxes"])
        print("Boxes shape:", target["boxes"].shape)
        print("Labels:", target["labels"])
        num_boxes = target["labels"].numel()
        print("Number of valid boxes:", num_boxes)
        print("Labels (unique):", target["labels"].unique() if num_boxes > 0 else "[]")

        break


if __name__ == "__main__":
    main()

