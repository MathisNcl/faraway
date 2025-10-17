from transformers import AutoImageProcessor, AutoModelForObjectDetection
from coco_dataset import CocoJSONDataset
import albumentations as A
import torch
from transformers import TrainingArguments, Trainer
import numpy as np
from dataclasses import dataclass
from transformers.image_transforms import center_to_corners_format
from torchmetrics.detection.mean_ap import MeanAveragePrecision

MODEL_NAME = "PekingU/rtdetr_v2_r18vd"
COCO_ANN_TRAIN_PATH = "/Users/mathisnicoli/Desktop/faraway_dataset/instances_train.json"
COCO_ANN_VAL_PATH = "/Users/mathisnicoli/Desktop/faraway_dataset/instances_val.json"
IMAGES_ROOT = "/Users/mathisnicoli/Desktop/faraway_dataset/images"
image_size = 640
image_processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME,
    do_resize=True,
    size={"width": image_size, "height": image_size},
    use_fast=True,
)

validation_transform = A.Compose(
    [A.NoOp()],
    bbox_params=A.BboxParams(
        format="coco",
        label_fields=["category"],
        clip=True,
        min_area=1,
        min_width=1,
        min_height=1,
    ),
)

train_dataset = CocoJSONDataset(
    COCO_ANN_TRAIN_PATH, IMAGES_ROOT, image_processor, transform=validation_transform
)
validation_dataset = CocoJSONDataset(
    COCO_ANN_VAL_PATH, IMAGES_ROOT, image_processor, transform=validation_transform
)

id2label = {cat["id"] - 1: cat["name"] for cat in train_dataset.coco_data["categories"]}
label2id = {v: k for k, v in id2label.items()}


def collate_fn(batch):
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    return data


@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


class MAPEvaluator:
    def __init__(self, image_processor, threshold=0.00, id2label=None):
        self.image_processor = image_processor
        self.threshold = threshold
        self.id2label = id2label

    def collect_image_sizes(self, targets):
        """Collect image sizes across the dataset as list of tensors with shape [batch_size, 2]."""
        image_sizes = []
        for batch in targets:
            batch_image_sizes = torch.tensor(np.array([x["size"] for x in batch]))
            image_sizes.append(batch_image_sizes)
        return image_sizes

    def collect_targets(self, targets, image_sizes):
        post_processed_targets = []
        for target_batch, image_size_batch in zip(targets, image_sizes):
            for target, size in zip(target_batch, image_size_batch):
                # here we have "yolo" format (x_center, y_center, width, height) in relative coordinates 0..1
                # and we need to convert it to "pascal" format (x_min, y_min, x_max, y_max) in absolute coordinates
                height, width = size
                boxes = torch.tensor(target["boxes"])
                boxes = center_to_corners_format(boxes)
                boxes = boxes * torch.tensor([[width, height, width, height]])

                labels = torch.tensor(target["class_labels"])
                post_processed_targets.append({"boxes": boxes, "labels": labels})
        return post_processed_targets

    def collect_predictions(self, predictions, image_sizes):
        post_processed_predictions = []
        for batch, target_sizes in zip(predictions, image_sizes):
            batch_logits, batch_boxes = batch[1], batch[2]
            output = ModelOutput(
                logits=torch.tensor(batch_logits), pred_boxes=torch.tensor(batch_boxes)
            )
            post_processed_output = self.image_processor.post_process_object_detection(
                output, threshold=self.threshold, target_sizes=target_sizes
            )
            post_processed_predictions.extend(post_processed_output)
        return post_processed_predictions

    @torch.no_grad()
    def __call__(self, evaluation_results):
        predictions, targets = (
            evaluation_results.predictions,
            evaluation_results.label_ids,
        )

        image_sizes = self.collect_image_sizes(targets)
        post_processed_targets = self.collect_targets(targets, image_sizes)
        post_processed_predictions = self.collect_predictions(predictions, image_sizes)

        evaluator = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
        evaluator.warn_on_many_detections = False
        evaluator.update(post_processed_predictions, post_processed_targets)

        metrics = evaluator.compute()

        # Replace list of per class metrics with separate metric for each class
        classes = metrics.pop("classes")
        map_per_class = metrics.pop("map_per_class")
        mar_100_per_class = metrics.pop("mar_100_per_class")
        for class_id, class_map, class_mar in zip(
            classes, map_per_class, mar_100_per_class
        ):
            class_name = (
                id2label[class_id.item()] if id2label is not None else class_id.item()
            )
            metrics[f"map_{class_name}"] = class_map
            metrics[f"mar_100_{class_name}"] = class_mar

        metrics = {k: round(v.item(), 4) for k, v in metrics.items()}

        return metrics


eval_compute_metrics_fn = MAPEvaluator(
    image_processor=image_processor, threshold=0.01, id2label=id2label
)

model = AutoModelForObjectDetection.from_pretrained(
    MODEL_NAME,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)


training_args = TrainingArguments(
    output_dir="rtdetr-v2-r18",
    num_train_epochs=40,
    max_grad_norm=0.1,
    learning_rate=5e-5,
    warmup_steps=50,
    per_device_train_batch_size=16,
    dataloader_num_workers=0,
    metric_for_best_model="eval_map",
    greater_is_better=True,
    load_best_model_at_end=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    remove_unused_columns=False,
    eval_do_concat_batches=False,
    report_to="tensorboard",
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    processing_class=image_processor,
    data_collator=collate_fn,
    compute_metrics=eval_compute_metrics_fn,
)

trainer.train()

trainer.save_model("rtdetr-v2-r18")
