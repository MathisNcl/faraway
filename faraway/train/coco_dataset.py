import json
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class CocoJSONDataset(Dataset):
    def __init__(
        self, coco_json_path: str, images_root: str, image_processor, transform=None
    ):
        with open(coco_json_path, "r") as f:
            self.coco_data = json.load(f)

        self.images_root = images_root
        self.image_processor = image_processor
        self.transform = transform

        # Mapping image_id -> image info
        self.images = {img["id"]: img for img in self.coco_data["images"]}

        # Mapping image_id -> list of annotations
        self.anns_per_image = {}
        for ann in self.coco_data["annotations"]:
            self.anns_per_image.setdefault(ann["image_id"], []).append(ann)

        # Build category id -> label index mapping
        cats = sorted(self.coco_data["categories"], key=lambda x: x["id"])
        self.catid2label = {c["id"]: idx for idx, c in enumerate(cats)}
        self.num_classes = len(self.catid2label)
        self.ids = list(self.images.keys())

    @staticmethod
    def format_image_annotations_as_coco(image_id, categories, boxes):
        annotations = []
        for category, bbox in zip(categories, boxes):
            annotations.append(
                {
                    "image_id": image_id,
                    "category_id": category,
                    "bbox": list(bbox),
                    "iscrowd": 0,
                    "area": bbox[2] * bbox[3],
                }
            )
        return {"image_id": image_id, "annotations": annotations}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        img_info = self.images[image_id]
        img_path = os.path.join(self.images_root, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        anns = self.anns_per_image.get(image_id, [])
        boxes = []
        categories = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, w, h])
            categories.append(self.catid2label[ann["category_id"]])

        # Apply augmentations if any
        if self.transform and len(boxes) > 0:
            transformed = self.transform(
                image=np.array(image), bboxes=boxes, category=categories
            )
            image = transformed["image"]
            boxes = transformed["bboxes"]
            categories = transformed["category"]

        # Format annotations for processor
        formatted_annotations = self.format_image_annotations_as_coco(
            image_id, categories, boxes
        )

        result = self.image_processor(
            images=image, annotations=formatted_annotations, return_tensors="pt"
        )
        result = {k: v[0] for k, v in result.items()}
        return result
