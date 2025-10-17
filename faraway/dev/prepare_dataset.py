import glob
from PIL import Image
from tqdm import tqdm

import torch
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
import json


class RTDETRPreAnnotator:
    def __init__(self, model_name="PekingU/rtdetr_v2_r18vd", confidence_threshold=0.5):
        """
        Initialise le modèle RT-DETR v2

        Args:
            model_name: Nom du modèle HuggingFace
            confidence_threshold: Seuil de confiance pour filtrer les détections
        """
        self.device = "mps" if torch.mps.is_available() else "cpu"
        print(f"Utilisation du device: {self.device}")

        self.processor = RTDetrImageProcessor.from_pretrained(model_name)
        self.model = RTDetrV2ForObjectDetection.from_pretrained(model_name).to(
            self.device
        )
        self.model.eval()
        self.confidence_threshold = confidence_threshold

    def detect_objects(self, image_path):
        """
        Détecte les objets dans une image

        Args:
            image_path: Chemin vers l'image

        Returns:
            Liste de détections avec boxes, scores et labels
        """
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-traitement
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([[height, width]]),
            threshold=self.confidence_threshold,
        )[0]

        detections = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            box = box.cpu().numpy()
            detections.append(
                {
                    "bbox": [
                        float(box[0]),
                        float(box[1]),
                        float(box[2]),
                        float(box[3]),
                    ],  # [x_min, y_min, x_max, y_max]
                    "score": float(score.cpu()),
                    "label": int(label.cpu()),
                    "label_name": self.model.config.id2label[int(label.cpu())],
                }
            )

        return detections, (width, height)

    def process_directory(self, image_files, output_json="annotations.json"):
        """
        Traite un répertoire d'images et génère les annotations

        Args:
            image_files: Liste des images à traiter
            output_json: Fichier JSON de sortie
        """

        annotations = {"images": [], "annotations": [], "categories": []}

        categories_set = set()

        print(f"Traitement de {len(image_files)} images...")
        annotation_id = 1

        for idx, image_file in enumerate(tqdm(image_files), start=1):
            try:
                detections, (width, height) = self.detect_objects(str(image_file))

                annotations["images"].append(
                    {
                        "id": idx,
                        "file_name": image_file.split("/")[-1],
                        "width": width,
                        "height": height,
                    }
                )

                # Ajouter les annotations
                for det in detections:
                    x_min, y_min, x_max, y_max = det["bbox"]
                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min

                    annotations["annotations"].append(
                        {
                            "id": annotation_id,
                            "image_id": idx,
                            "category_id": 6129114,
                            "bbox": [x_min, y_min, bbox_width, bbox_height],
                            "area": bbox_width * bbox_height,
                            "score": det["score"],
                            "iscrowd": 0,
                            "segmentation": [],
                        }
                    )

                    categories_set.add((det["label"], det["label_name"]))
                    annotation_id += 1

            except Exception as e:
                print(f"Erreur lors du traitement de {image_file}: {e}")

        # Ajouter les catégories
        # for cat_id, cat_name in sorted(categories_set):
        #     annotations["categories"].append({"id": cat_id, "name": cat_name})
        annotations["categories"].append({"id": 6129114, "name": "exploration"})

        # Sauvegarder le JSON
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)

        print(f"\nAnnotations sauvegardées dans {output_json}")
        print(f"Total images: {len(annotations['images'])}")
        print(f"Total annotations: {len(annotations['annotations'])}")
        print(f"Catégories trouvées: {len(annotations['categories'])}")


if __name__ == "__main__":
    paths = glob.glob("/Users/mathisnicoli/Desktop/faraway_dataset/preprocessed/*.jpg")
    paths.sort()
    # cpt = 0
    # prefix = "exploration_"
    # for path in tqdm(paths):
    #     name = path.split("/")[-1]
    #     img = Image.open(path).resize((768, 1024))
    #     if name == "IMG20251016101508.jpg":
    #         cpt = 0
    #         prefix = "sanctuary_"
    #     img.save(f"/Users/mathisnicoli/Desktop/faraway_dataset/preprocessed/{prefix}{cpt}.jpg")
    #     cpt += 1

    # Configuration
    OUTPUT_JSON = "annotations_coco.json"
    CONFIDENCE_THRESHOLD = 0.6

    # Initialiser et lancer
    annotator = RTDETRPreAnnotator(
        model_name="PekingU/rtdetr_v2_r18vd", confidence_threshold=CONFIDENCE_THRESHOLD
    )

    annotator.process_directory(paths, OUTPUT_JSON)
