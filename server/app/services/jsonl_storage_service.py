import json
import os
from typing import Dict, List, Optional

from .storage_service import StorageService, normalize_plant, normalize_trial


class JsonlStorageService(StorageService):
    """Append-only JSONL backend.

    Mirrors the original storage layout: ``<plants_dir>/plants.json`` and
    ``<trials_dir>/trials.json`` each contain one JSON object per line. Saves
    append a new line; loads collapse to the latest entry per id. Useful for
    backwards compatibility and as a migration source.
    """

    def __init__(self, plants_dir: str = "plants", trials_dir: str = "trials"):
        self.plants_dir = plants_dir
        self.trials_dir = trials_dir
        os.makedirs(self.plants_dir, exist_ok=True)
        os.makedirs(self.trials_dir, exist_ok=True)

        self.plants_file = os.path.join(self.plants_dir, "plants.json")
        self.trials_file = os.path.join(self.trials_dir, "trials.json")

        for path in (self.plants_file, self.trials_file):
            if not os.path.exists(path):
                open(path, "w").close()

    def load_plants(self) -> Dict[str, dict]:
        plants: Dict[str, dict] = {}
        with open(self.plants_file, "r") as fs:
            for line in fs:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                plants[data["plant_id"]] = normalize_plant(data)
        return plants

    def save_plant(self, plant: dict) -> None:
        with open(self.plants_file, "a") as fs:
            fs.write(json.dumps(normalize_plant(plant)) + "\n")

    def get_plant(self, plant_id: str) -> Optional[dict]:
        return self.load_plants().get(plant_id)

    def list_plant_ids(self) -> List[str]:
        return list(self.load_plants().keys())

    def load_trials(self) -> Dict[str, dict]:
        trials: Dict[str, dict] = {}
        with open(self.trials_file, "r") as fs:
            for line in fs:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                trials[data["trial_id"]] = normalize_trial(data)
        return trials

    def save_trial(self, trial: dict) -> None:
        with open(self.trials_file, "a") as fs:
            fs.write(json.dumps(normalize_trial(trial)) + "\n")
