import json
from pathlib import Path


DATA_DIR = Path(__file__).parent / "data"

INPUT_FILE = DATA_DIR / "default_config_v2.json"
OUTPUT_FILE = DATA_DIR / "default_config_v3.json"


SCENARIOS = [
    "Domination",
    "Capture & Control",
    "Breakthrough",
    "Stake a Claim",
    "To the Death!",
    "Lords of Battle",
    "Assassination",
    "Contest of Champions",
    "Hold Ground",
    "Heirloom of Ages Past",
    "Sites of Power",
    "Command the Battlefield",
    "Destroy the Supplies",
    "Retrieval",
    "Seize the Prizes",
    "Treasure Hoard",
    "Reconnoitre",
    "Storm the Camp",
    "Divide & Conquer",
    "Escort the Wounded",
    "Fog of War",
    "Clash by Moonlight",
    "Lead from the Front",
    "Convergence",
]


def main() -> None:
    with INPUT_FILE.open("r", encoding="utf-8") as file:
        config = json.load(file)

    new_ratings = {}

    for army, data in config["ratings"].items():
        scenario_modifiers = {
            scenario: data[scenario]
            for scenario in SCENARIOS
        }

        new_ratings[army] = {
            "alignment": data["alignment"],
            "base_score": data["base_score"],
            "scenario_modifiers": scenario_modifiers,
        }

    new_config = {
        "metadata": {
            **config.get("metadata", {}),
            "version": 3,
            "rating_model": (
                "Base army score 0.45-0.55 "
                "+ scenario modifier -0.20 to +0.20"
            ),
        },
        "ratings": new_ratings,
    }

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            new_config,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Created {OUTPUT_FILE}")


if __name__ == "__main__":
    main()