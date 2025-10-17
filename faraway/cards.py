import pandas as pd
import numpy as np
from PIL import Image
from collections import Counter


class SanctuaryCard:
    def __init__(self, image: Image.Image):
        self.color = self._get_color(image)
        self.material = self._get_material(image)
        self.has_hint = self._get_hint(image)
        self.is_night = self._get_night(image)

    @staticmethod
    def _get_color(image: Image.Image):
        img_hsv = image.convert("HSV")
        pixels = np.array(img_hsv)
        pixels = pixels.reshape(-1, 3)

        count = Counter()
        for pixel in pixels:
            h, s, v = pixel

            # grey
            if s < 50:
                count["grey"] += 1
            # red (0-15 ou 345-360)
            elif h < 15 or h > 230:
                count["red"] += 1
            # yellow (15-75)
            elif 15 <= h < 75:
                count["yellow"] += 1
            # green (75-165)
            elif 75 <= h < 165:
                count["green"] += 1
            # blue (165-230)
            elif 165 <= h < 230:
                count["bleu"] += 1

        return count.most_common(1)[0][0]

    def _get_night(self, image: Image.Image):
        pass

    def _get_hint(self, image: Image.Image):
        pass

    def _get_material(self, image: Image.Image):
        pass


class SanctuaryCardDummy:
    def __init__(self, color, material, has_hint, is_night, quest_trigger, quest_point):
        self.color = color
        self.material = material
        self.has_hint = has_hint
        self.is_night = is_night
        self.quest_trigger = quest_trigger
        self.quest_point = quest_point


class ExplorationCard:
    def __init__(self, id: int, all_cards: pd.DataFrame):
        infos = all_cards.loc[id]
        self.id = id
        self.has_hint = bool(infos.hint > 0)
        self.is_night = bool(infos.night > 0)
        self.nb_stones = self._get_value_or_0(infos.stone)
        self.nb_heads = self._get_value_or_0(infos["head"])
        self.nb_plants = self._get_value_or_0(infos.plant)
        self.color = infos.color

        # Quests
        self.quest_needs = {"stone": 0, "head": 0, "plant": 0}

        for i in range(1, 6):
            need = infos[f"quest_need_{i}"]
            if isinstance(need, str) is False and np.isnan(need):
                break
            self.quest_needs[need] += 1

        self.quest_trigger = infos["quest_win"]
        self.quest_point = self._get_value_or_0(infos["quest_point"])

    @staticmethod
    def _get_value_or_0(value):
        return int(value) if bool(np.isnan(value)) is False else 0

    def __repr__(self):
        return f"""Card(id={self.id}, stones={self.nb_stones}, heads={self.nb_heads}, plants={self.nb_plants}, 
        color={self.color}, night={self.is_night}, hint={self.has_hint}, quest_needs={self.quest_needs}, 
        quest_trigger={self.quest_trigger}, quest_point={self.quest_point})"""


class Board:
    def __init__(self, explorations: list[ExplorationCard], sanctuaries: list[SanctuaryCard]):
        self.explorations = explorations[::-1]
        self.sanctuaries = sanctuaries
        self.aggregated_sanctuaries = self.aggregate_sanctuaries()

    def aggregate_sanctuaries(self) -> dict[str, int]:
        aggregated_sanctuaries = {
            "hint": 0,
            "stone": 0,
            "head": 0,
            "plant": 0,
            "red": 0,
            "yellow": 0,
            "green": 0,
            "blue": 0,
            "night": 0,
        }

        for sanctuary in self.sanctuaries:
            aggregated_sanctuaries["hint"] += sanctuary.has_hint
            if sanctuary.material is not None:
                aggregated_sanctuaries[sanctuary.material] += 1
            if sanctuary.color != "grey":
                aggregated_sanctuaries[sanctuary.color] += 1
            aggregated_sanctuaries["night"] += sanctuary.is_night

        return aggregated_sanctuaries

    def compute_final_score(self):
        score = 0
        detailled_score = {}
        round_situation = self.aggregated_sanctuaries.copy()

        for round_i, exploration in enumerate(self.explorations, start=1):
            # add caracteristics to the situation aggregated
            round_situation["stone"] += exploration.nb_stones
            round_situation["head"] += exploration.nb_heads
            round_situation["plant"] += exploration.nb_plants
            round_situation["night"] += exploration.is_night
            round_situation["hint"] += exploration.has_hint
            round_situation[exploration.color] += 1

            # check if the quest is succeed
            # 1. if no quest but some points free
            need_some_material: bool = sum(exploration.quest_needs.values()) != 0
            detailled_score[f"round_{round_i}"] = 0
            quest_succeed: bool = False
            if need_some_material is False and exploration.quest_point > 0:
                quest_succeed = True
            # 2. There is a quest
            else:
                enough_material_to_succed_quest: dict[str, bool] = {
                    "stone": False,
                    "head": False,
                    "plant": False,
                }
                for material, need in exploration.quest_needs.items():
                    if round_situation[material] >= need:
                        enough_material_to_succed_quest[material] = True
                quest_succeed: bool = all(enough_material_to_succed_quest.values())

            if quest_succeed is False:
                continue
            # If no trigger, get the points
            if isinstance(exploration.quest_trigger, str) is False:
                detailled_score[f"round_{round_i}"] += exploration.quest_point
            else:
                # extract the quest trigger and count
                detailled_score = self.extract_quest_trigger_and_count(
                    card=exploration,
                    detailled_score=detailled_score,
                    key=f"round_{round_i}",
                    round_situation=round_situation,
                )

            score += detailled_score[f"round_{round_i}"]

        # Count sanctury
        detailled_score["sanctuary"] = 0
        for sanctuary in self.sanctuaries:
            # nothing
            if sanctuary.quest_point == 0:
                continue
            # only free points
            if sanctuary.quest_trigger is None:
                detailled_score["sanctuary"] += sanctuary.quest_point
            else:
                detailled_score = self.extract_quest_trigger_and_count(
                    card=sanctuary,
                    detailled_score=detailled_score,
                    key="sanctuary",
                    round_situation=round_situation,
                )

        score += detailled_score["sanctuary"]

        return detailled_score, score

    @staticmethod
    def extract_quest_trigger_and_count(
        card: ExplorationCard | SanctuaryCard,
        detailled_score,
        key: str,
        round_situation: dict[str, int],
    ):
        for col_mat in [
            "red",
            "yellow",
            "green",
            "blue",
            "night",
            "hint",
            "stone",
            "head",
            "plant",
        ]:
            if col_mat in card.quest_trigger:
                detailled_score[key] += card.quest_point * round_situation[col_mat]
        if card.quest_trigger == "all colors":
            detailled_score[key] += card.quest_point * min(
                round_situation["red"],
                round_situation["yellow"],
                round_situation["green"],
                round_situation["blue"],
            )
        return detailled_score


# all_cards: pd.DataFrame = pd.read_csv(
#     "/Users/mathisnicoli/Desktop/Projets/faraway/faraway/dev/faraway_cards.csv", sep=";"
# ).set_index("id")

# explorations: list[ExplorationCard] = [
#     ExplorationCard(id, all_cards) for position, id in enumerate([18, 25, 63, 30, 22, 66, 19, 29])
# ]
# sanctuaries = [
#     SanctuaryCardDummy("grey", None, False, False, "head", 2),
#     SanctuaryCardDummy("red", "head", False, False, None, 0),
#     SanctuaryCardDummy("grey", None, False, False, "stone", 2),
#     SanctuaryCardDummy("yellow", None, True, False, None, 0),
#     SanctuaryCardDummy("yellow", None, False, False, "all colors", 4),
#     SanctuaryCardDummy("grey", "stone", False, True, None, 0),
#     SanctuaryCardDummy("grey", "stone", False, False, "hint", 1),
# ]

# explorations: list[ExplorationCard] = [
#     ExplorationCard(id, all_cards) for position, id in enumerate([46, 45, 63, 35, 28, 16, 15, 9])
# ]
# sanctuaries = [
#     SanctuaryCardDummy("grey", "head", True, False, None, 0),
#     SanctuaryCardDummy("blue", "head", False, False, None, 0),
#     SanctuaryCardDummy("grey", "plant", True, False, None, 0),
#     SanctuaryCardDummy("green", "head", False, False, None, 0),
#     SanctuaryCardDummy("blue", None, False, False, "blue", 1),
#     SanctuaryCardDummy("yellow", None, False, False, "hint", 1),
#     SanctuaryCardDummy("grey", None, False, False, "head", 2),
# ]
# board = Board(explorations, sanctuaries)


# print("Score: ", board.compute_final_score())
