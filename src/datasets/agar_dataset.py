import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


class AGARDataset(Dataset):
    def __init__(self, img_dir, ann_dir, transforms=None,
                 patch_size=512, stride=512, include_empty=True):
        """
        img_dir: folder with images (.jpg/.png)
        ann_dir: folder with per-image JSON annotation files
        transforms: torchvision v2 transforms (applied to image and target)
        patch_size: size of cropped patches (square)
        stride: step size between crops (can be < patch_size for overlap)
        include_empty: whether to keep patches with no colonies
        """
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transforms = transforms
        self.patch_size = patch_size
        self.stride = stride
        self.include_empty = include_empty

        # collect all image IDs (filenames without extensions)
        self.ids = [
            os.path.splitext(f)[0]
            for f in os.listdir(self.img_dir)
            if f.lower().endswith((".jpg", ".png"))
        ]
        self.ids.sort()

        # precompute patch index map
        self.index_map = []
        for img_idx, sample_id in enumerate(self.ids):
            img_path = os.path.join(self.img_dir, sample_id + ".jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(self.img_dir, sample_id + ".png")
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            for top in range(0, h, stride):
                for left in range(0, w, stride):
                    bottom = min(top + patch_size, h)
                    right = min(left + patch_size, w)
                    self.index_map.append((img_idx, left, top, right, bottom))


    def __len__(self):
        return len(self.index_map)
    

    def __getitem__(self, idx):
        img_idx, left, top, right, bottom = self.index_map[idx]
        sample_id = self.ids[img_idx]

        img_path = os.path.join(self.img_dir, sample_id + ".jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir, sample_id + ".png")
        ann_path = os.path.join(self.ann_dir, sample_id + ".json")

        # load image and crop
        full_img = Image.open(img_path).convert("RGB")
        img = full_img.crop((left, top, right, bottom))
        img = tv_tensors.Image(img)
        
        # load json annotations 
        with open(ann_path, "r") as f:
            ann = json.load(f)

        boxes, labels = [], []
        if ann.get("colonies_number", 0) > 0 and "labels" in ann:
            for obj in ann["labels"]:
                x_min, y_min = obj["x"], obj["y"]
                x_max, y_max = x_min + obj["width"], y_min + obj["height"]

                # keep boxes overlapping with patch
                if x_max > left and x_min < right and y_max > top and y_min < bottom:
                    x_min = max(x_min - left, 0)
                    y_min = max(y_min - top, 0)
                    x_max = min(x_max - left, right - left)
                    y_max = min(y_max - top, bottom - top)
                    if x_max > x_min and y_max > y_min:
                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(1)  # single class: colony

        # skip empty patches if include_empty=False
        if len(boxes) == 0 and not self.include_empty:
            return self.__getitem__((idx + 1) % len(self))

        # convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img)),
            "labels": labels,
            "image_id": torch.tensor([img_idx]),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if len(boxes) > 0 else torch.tensor([]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target
