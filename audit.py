import os
import json
from pathlib import Path

# === CONFIG ===
IMG_DIR = Path("data/images")  # path to your images/ folder
ANN_FILE = Path("data/annotations.json")  # path to your COCO annotations file

# === LOAD JSON ===
with open(ANN_FILE, "r") as f:
    coco = json.load(f)

# Collect image file names and IDs
json_images = {img["file_name"]: (img["id"], img["width"], img["height"]) for img in coco["images"]}
json_image_ids = {img["id"] for img in coco["images"]}

# Collect actual files in images/ folder
folder_images = {f.name for f in IMG_DIR.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]}

# === Checks ===
missing_in_json = folder_images - set(json_images.keys())
missing_in_folder = set(json_images.keys()) - folder_images

invalid_annotation_ids = [ann["image_id"] for ann in coco["annotations"] if ann["image_id"] not in json_image_ids]

# === Bounding box checks ===
invalid_boxes = []
for ann in coco["annotations"]:
    img_id = ann["image_id"]
    if img_id not in json_images:
        continue
    _, img_w, img_h = json_images[next(k for k, v in json_images.items() if v[0] == img_id)]
    x, y, w, h = ann["bbox"]

    if w <= 0 or h <= 0:
        invalid_boxes.append((ann["id"], "non-positive size", ann["bbox"]))
        continue
    if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
        invalid_boxes.append((ann["id"], "out of bounds", ann["bbox"]))

# === Report ===
print("=== Audit Report ===")
print(f"Total images in folder: {len(folder_images)}")
print(f"Total images in JSON:   {len(json_images)}")
print(f"Total annotations:      {len(coco['annotations'])}")

if missing_in_json:
    print(f"\nImages in folder but missing in JSON ({len(missing_in_json)}):")
    for f in sorted(list(missing_in_json))[:10]:
        print("  ", f)

if missing_in_folder:
    print(f"\nImages in JSON but missing in folder ({len(missing_in_folder)}):")
    for f in sorted(list(missing_in_folder))[:10]:
        print("  ", f)

if invalid_annotation_ids:
    print(f"\nAnnotations pointing to missing image ids ({len(invalid_annotation_ids)}):")
    print("  Sample:", invalid_annotation_ids[:10])

if invalid_boxes:
    print(f"\nInvalid bounding boxes ({len(invalid_boxes)}):")
    for b in invalid_boxes[:10]:
        print("  ", b)

if not (missing_in_json or missing_in_folder or invalid_annotation_ids or invalid_boxes):
    print("\nAll good! Images and annotations line up, and bboxes are valid.")

