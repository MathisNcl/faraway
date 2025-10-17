import os
import json
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import albumentations as A
import cv2
from pycocotools.coco import COCO

# -----------------------------
# CONFIG
# -----------------------------
IMAGES_DIR = "/Users/mathisnicoli/Desktop/faraway_dataset/preprocessed"
COCO_JSON = "/Users/mathisnicoli/Desktop/faraway_dataset/annotations_coco_v1.json"
OUTPUT_DIR = "/Users/mathisnicoli/Desktop/faraway_dataset/"
AUG_RATIO = 1.0  # 1.0 -> as many augmented image as original (so x2 more images)
VAL_RATIO = 0.2  # 80/20 split
SEED = 42
# -----------------------------

os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)

random.seed(SEED)

# loading COCO
coco = COCO(COCO_JSON)
img_ids = list(coco.imgs.keys())
cat_ids = coco.getCatIds()

# Transformation Albumentations
transform = A.Compose(
    [
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.6),
        A.RandomBrightnessContrast(p=0.5),
    ],
    bbox_params=A.BboxParams(
        format="coco", label_fields=["category_id"], min_visibility=0.2, clip=True
    ),
)

# New coco structure
new_images = []
new_annotations = []
new_img_id = 1
new_ann_id = 1

for img_id in tqdm(img_ids, desc="Augmentation"):
    img_info = coco.loadImgs(img_id)[0]
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=None))
    img_path = os.path.join(IMAGES_DIR, img_info["file_name"])

    image = cv2.imread(img_path)
    if image is None:
        continue

    # save original image
    base_name = os.path.splitext(os.path.basename(img_info["file_name"]))[0]
    orig_file = f"{base_name}_orig.jpg"
    cv2.imwrite(os.path.join(OUTPUT_DIR, "images", orig_file), image)

    # add to json
    new_images.append(
        {
            "id": new_img_id,
            "file_name": orig_file,
            "width": image.shape[1],
            "height": image.shape[0],
        }
    )
    for a in anns:
        new_annotations.append(
            {
                "id": new_ann_id,
                "image_id": new_img_id,
                "category_id": a["category_id"],
                "bbox": a["bbox"],
                "area": a["area"],
                "iscrowd": a["iscrowd"],
            }
        )
        new_ann_id += 1
    new_img_id += 1

    # Augmentation
    n_aug = int(AUG_RATIO)
    for i in range(n_aug):
        bboxes = [a["bbox"] for a in anns]
        cat_ids_ann = [a["category_id"] for a in anns]
        transformed = transform(image=image, bboxes=bboxes, category_id=cat_ids_ann)

        aug_img = transformed["image"]
        aug_bboxes = transformed["bboxes"]
        aug_cats = transformed["category_id"]

        # save
        aug_filename = f"{base_name}_aug{i}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_DIR, "images", aug_filename), aug_img)

        # add to json
        new_images.append(
            {
                "id": new_img_id,
                "file_name": aug_filename,
                "width": aug_img.shape[1],
                "height": aug_img.shape[0],
            }
        )

        # new annotations
        for bbox, cat in zip(aug_bboxes, aug_cats):
            x, y, w, h = bbox
            area = w * h
            new_annotations.append(
                {
                    "id": new_ann_id,
                    "image_id": new_img_id,
                    "category_id": cat,
                    "bbox": [x, y, w, h],
                    "area": area,
                    "iscrowd": 0,
                }
            )
            new_ann_id += 1

        new_img_id += 1

# same cats
categories = coco.loadCats(cat_ids)

# full dataset
full_data = {
    "images": new_images,
    "annotations": new_annotations,
    "categories": categories,
}

# Split train / val
image_ids = [img["id"] for img in new_images]
train_ids, val_ids = train_test_split(image_ids, test_size=VAL_RATIO, random_state=SEED)


def split_json(ids, data):
    imgs = [img for img in data["images"] if img["id"] in ids]
    anns = [ann for ann in data["annotations"] if ann["image_id"] in ids]
    return {"images": imgs, "annotations": anns, "categories": data["categories"]}


train_json = split_json(train_ids, full_data)
val_json = split_json(val_ids, full_data)

# save json files
with open(os.path.join(OUTPUT_DIR, "instances_train.json"), "w") as f:
    json.dump(train_json, f)
with open(os.path.join(OUTPUT_DIR, "instances_val.json"), "w") as f:
    json.dump(val_json, f)

print(
    f"Done. Train: {len(train_json['images'])} images, Val: {len(val_json['images'])} images"
)
print(f"Save at: {OUTPUT_DIR}")
