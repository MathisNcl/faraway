# take the filtered predictions and concert it into Board and Cards
import numpy as np
import supervision as sv
from paddleocr import PaddleOCR
from PIL import Image
from cards import ExplorationCard
import re
import pandas as pd


class PostProcessor:
    def __init__(self, all_cards_path: str) -> None:
        self.ocr_client: PaddleOCR = PaddleOCR(
            use_angle_cls=True,
            lang="fr",
        )
        self.all_cards: pd.DataFrame = pd.read_csv(all_cards_path, sep=";").set_index("id")

    def _extract_and_sort_explorations(self, detections: sv.Detections) -> sv.Detections:
        """
        Order detections as a grid:
        0. Get explorations
        1. Sort by x (left to right)
        2. Split in 2 group based on y (top/bottom)
        3. Concatenate : upper lines then lower lines
        """
        # filter
        detections = detections[detections.class_id == 0]

        # compute centers
        centers = np.column_stack(
            [
                (detections.xyxy[:, 1] + detections.xyxy[:, 3]) / 2,  # centre y
                (detections.xyxy[:, 0] + detections.xyxy[:, 2]) / 2,  # centre x
            ]
        )

        # Sort by x
        x_sorted_indices = np.argsort(centers[:, 1])

        # Split in 2 groups
        upper_line = []
        lower_line = []

        for i in range(0, len(x_sorted_indices), 2):
            idx1 = x_sorted_indices[i]
            idx2 = x_sorted_indices[i + 1] if i + 1 < len(x_sorted_indices) else None

            if idx2 is not None:
                # Compare y, lower is going to top
                if centers[idx1, 0] < centers[idx2, 0]:
                    upper_line.append(idx1)
                    lower_line.append(idx2)
                else:
                    upper_line.append(idx2)
                    lower_line.append(idx1)
            else:
                upper_line.append(idx1)

        # Concat upper line and lower line
        sorted_indices = np.array(upper_line + lower_line)

        # Format
        sorted_detections = sv.Detections(
            xyxy=detections.xyxy[sorted_indices],
            mask=detections.mask[sorted_indices] if detections.mask is not None else None,
            confidence=detections.confidence[sorted_indices] if detections.confidence is not None else None,
            class_id=detections.class_id[sorted_indices] if detections.class_id is not None else None,
            tracker_id=detections.tracker_id[sorted_indices] if detections.tracker_id is not None else None,
            data={k: v[sorted_indices] for k, v in detections.data.items()} if detections.data else {},
            metadata=detections.metadata,
        )

        return sorted_detections

    def get_exploration_cards(self, image: Image.Image, detections: sv.Detections) -> list[ExplorationCard]:
        exploration_detections = self._extract_and_sort_explorations(detections)
        exploration_cards: list[ExplorationCard] = []
        # for every detection, get the top card, use OCR and retrieve the Card object
        for detection in exploration_detections:
            found: bool = False
            cropped_image: Image.Image = image.crop(detection[0])
            for ratio in [2, 1]:
                if found:
                    break
                top_card: Image.Image = cropped_image.crop(
                    (0, 0, cropped_image.width // ratio, cropped_image.height // 2)
                )
                ocr: list[dict] = self.ocr_client.predict(np.array(top_card.convert("RGB")))
                for text, score in zip(ocr[0]["rec_texts"], ocr[0]["rec_scores"]):
                    if re.match(r"\d{1,2}", text) and score > 0.7:
                        # remove trailing 0
                        text = text if len(text) <= 2 else text[:-1]
                        exploration_cards.append(ExplorationCard(int(text), self.all_cards))
                        found = True
                        break
            if found is False:
                # display(top_card)
                raise ValueError("Card not found in ocr:\n{}".format(ocr[0]["rec_texts"]))
        return exploration_cards
