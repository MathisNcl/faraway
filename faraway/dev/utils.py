from PIL import Image
import supervision as sv


def show_inference(image: Image.Image, results: list | sv.Detections) -> Image.Image:
    detections = results
    if isinstance(results, sv.Detections) is False:
        detections = sv.Detections.from_transformers(results[0])
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_scale=0.8)

    annotated_frame = box_annotator.annotate(
        scene=image.copy(),
        detections=detections,
    )
    labels = [
        f"{class_id} {confidence:0.2f}"
        for box, _, confidence, class_id, _, _ in detections
    ]
    labels = [f"idx_{idx}, {label}" for idx, label in enumerate(labels)]
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels,
    )

    return annotated_frame
