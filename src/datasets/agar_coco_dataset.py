import torch
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from PIL import Image
import warnings
import os


class AGARCocoDataset(Dataset):
    def __init__(self, img_dir, ann_file, transforms=None,
                 patch_size=512, stride=512, include_empty=True):
        """
        COCO-style dataset with patching support.

        Args:
            img_dir (str): path to images folder
            ann_file (str): path to COCO annotations.json
            transforms: torchvision v2 transforms
            patch_size (int): size of cropped patches
            stride (int): step between crops (overlap if < patch_size)
            include_empty (bool): keep empty patches or skip them
        """
        self.coco = CocoDetection(img_dir, ann_file)
        self.img_dir = img_dir
        self.transforms = transforms
        self.patch_size = patch_size
        self.stride = stride
        self.include_empty = include_empty

        # Collect all image info
        self.ids = list(self.coco.ids)

        # Precompute patch index map
        self.index_map = []
        for img_idx, img_id in enumerate(self.ids):
            img_info = self.coco.coco.loadImgs(img_id)[0]
            w, h = img_info["width"], img_info["height"]
            for top in range(0, h, stride):
                for left in range(0, w, stride):
                    bottom = min(top + patch_size, h)
                    right = min(left + patch_size, w)
                    self.index_map.append((img_idx, left, top, right, bottom))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        img_idx, left, top, right, bottom = self.index_map[idx]
        img_id = self.ids[img_idx]

        # Load full image + annotations
        try:
            full_img, anns = self.coco[img_idx]
        except FileNotFoundError as e:
            warnings.warn(f"Skipping missing file: {e.filename}")
            return self.__getitem__((idx + 1) % len(self))

        # Crop patch
        patch = full_img.crop((left, top, right, bottom))
        patch = tv_tensors.Image(patch).float() / 255.0

        # Collect boxes inside this patch
        boxes, labels, areas, iscrowd = [], [], [], []
        for obj in anns:
            x, y, w, h = obj["bbox"]

            # Skip invalid boxes
            if w <= 0 or h <= 0:
                warnings.warn(f"Skipping invalid bbox {obj['bbox']} (img_id={img_id})")
                continue
            if x < 0 or y < 0 or x + w > full_img.width or y + h > full_img.height:
                warnings.warn(f"Skipping out-of-bounds bbox {obj['bbox']} (img_id={img_id})")
                continue

            x_min, y_min, x_max, y_max = x, y, x + w, y + h

            # Keep only boxes overlapping with the patch
            if x_max > left and x_min < right and y_max > top and y_min < bottom:
                # Shift coords into patch space
                x_min = max(x_min - left, 0)
                y_min = max(y_min - top, 0)
                x_max = min(x_max - left, right - left)
                y_max = min(y_max - top, bottom - top)

                if x_max > x_min and y_max > y_min:
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(1)  # single-class setup
                    areas.append((x_max - x_min) * (y_max - y_min))
                    iscrowd.append(obj.get("iscrowd", 0))

        # Skip empty patches if requested
        if len(boxes) == 0 and not self.include_empty:
            return self.__getitem__((idx + 1) % len(self))

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,), dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(patch)),
            "labels": labels,
            "area": areas,
            "iscrowd": iscrowd,
            "image_id": torch.tensor([img_id]),
            # === NEW fields for plate-level eval ===
            "patch_origin": (left, top),                        # patch top-left corner in plate coords
            "orig_size": (full_img.height, full_img.width),     # (H,W) of full plate
        }

        if self.transforms is not None:
            patch, target = self.transforms(patch, target)

        return patch, target

