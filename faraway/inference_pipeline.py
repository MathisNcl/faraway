from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import supervision as sv
import numpy as np
import torch
from post_processor import PostProcessor


class InferencePipeline:
    def __init__(self, model_path: str, csv_path: str, device: str = "cpu") -> None:
        self.image_processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = AutoModelForObjectDetection.from_pretrained(model_path)
        self.model = self.model.to(device)
        self.model.eval()
        print("Modèle chargé et prêt pour l'inférence.")
        self.post_processor = PostProcessor(csv_path)

    def infer_on_image(self, image: str | Image.Image, threshold=0.5) -> sv.Detections:
        if isinstance(image, str):
            image = Image.open(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        inputs = self.image_processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.image_processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([(image.height, image.width)]),
            threshold=threshold,
        )
        detections = sv.Detections.from_transformers(results[0])

        return detections

    @staticmethod
    def compute_area(bbox: list[float]) -> float:
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    @staticmethod
    def compute_ratio(bbox: list[float]) -> float:
        return (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])

    def filter_detections(
        self,
        detections: sv.Detections,
        threshold_iou: float = 0.7,
        threshold_ratio: float = 0.8,
    ):
        filtered_detections: list[tuple] = []
        for i, detection_i in enumerate(detections):
            keep: bool = True
            for j, detection_j in enumerate(detections):
                # xyxy is at index 0
                iou: float = sv.box_iou(detection_i[0], detection_j[0])
                if i != j and iou > threshold_iou:
                    area_i: float = self.compute_area(detection_i[0])
                    area_j: float = self.compute_area(detection_j[0])

                    # Garder celle avec la plus grande area
                    # En cas d'égalité, garder celle avec le plus petit indice
                    if area_i < area_j or (area_i == area_j and i > j):
                        keep = False
                        break
            if keep:
                filtered_detections.append(detection_i)

        # set it back to detections format
        detections_dict = {
            "xyxy": np.array([list(det[0]) for det in filtered_detections]),
            "confidence": np.array([float(det[2]) for det in filtered_detections]),
            "class_id": np.array(
                [
                    1 if self.compute_ratio(det[0]) < threshold_ratio else 0
                    for det in filtered_detections
                ]
            ),
            "mask": None,
            "tracker_id": None,
            "data": {},
            "metadata": {},
        }
        return sv.Detections(**detections_dict)

    def get_cards_on_image(
        self,
        image: str | Image.Image,
        threshold=0.5,
        threshold_iou=0.7,
        threshold_ratio=0.8,
    ) -> sv.Detections:
        detections: sv.Detections = self.infer_on_image(image, threshold)
        filtered_detections: sv.Detections = self.filter_detections(
            detections, threshold_iou, threshold_ratio
        )
        return filtered_detections
