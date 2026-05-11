import json
import os
from shutil import rmtree

import pytest

from services import JsonlStorageService


@pytest.fixture
def tmp_storage(tmp_path):
    plants = tmp_path / "plants"
    trials = tmp_path / "trials"
    storage = JsonlStorageService(plants_dir=str(plants), trials_dir=str(trials))
    yield storage, plants, trials


def test_init_creates_files(tmp_storage):
    _, plants, trials = tmp_storage
    assert os.path.exists(plants / "plants.json")
    assert os.path.exists(trials / "trials.json")


def test_save_and_load_plant_round_trip(tmp_storage):
    storage, _, _ = tmp_storage
    storage.save_plant({
        "plant_id": "abc",
        "status": "complete",
        "image": "abc.jpeg",
        "bounding_boxes": [[0.1, 0.2, 0.3, 0.4]],
        "masks": [[[0.0, 0.0], [1.0, 1.0]]],
        "labels": ["Healthy-Leaf"],
    })

    plants = storage.load_plants()
    assert plants["abc"]["status"] == "complete"
    assert plants["abc"]["bounding_boxes"] == [[0.1, 0.2, 0.3, 0.4]]
    assert plants["abc"]["labels"] == ["Healthy-Leaf"]
    assert plants["abc"]["trial_id"] == ""  # default-filled


def test_save_plant_appends_so_latest_wins(tmp_storage):
    storage, plants_dir, _ = tmp_storage
    storage.save_plant({"plant_id": "x", "status": "pending", "image": "x.jpeg"})
    storage.save_plant({"plant_id": "x", "status": "complete", "image": "x.jpeg"})

    # Two physical lines (append-only)
    with open(plants_dir / "plants.json") as fs:
        lines = [line for line in fs if line.strip()]
    assert len(lines) == 2

    # But load collapses to the latest
    assert storage.load_plants()["x"]["status"] == "complete"


def test_get_plant_returns_none_for_missing(tmp_storage):
    storage, _, _ = tmp_storage
    assert storage.get_plant("missing") is None


def test_list_plant_ids_preserves_insertion_order(tmp_storage):
    storage, _, _ = tmp_storage
    for pid in ["a", "b", "c"]:
        storage.save_plant({"plant_id": pid, "status": "pending", "image": f"{pid}.jpeg"})
    assert storage.list_plant_ids() == ["a", "b", "c"]


def test_save_and_load_trial(tmp_storage):
    storage, _, _ = tmp_storage
    storage.save_trial({
        "trial_id": "t1",
        "trial_name": "North Field",
        "description": "rust resistance",
    })

    trials = storage.load_trials()
    assert trials["t1"]["trial_name"] == "North Field"
    assert trials["t1"]["description"] == "rust resistance"
    assert trials["t1"]["user"] == ""


def test_load_skips_blank_lines(tmp_path):
    plants = tmp_path / "plants"
    trials = tmp_path / "trials"
    plants.mkdir()
    trials.mkdir()
    (plants / "plants.json").write_text(
        json.dumps({"plant_id": "p", "status": "complete", "image": "p.jpeg"}) + "\n\n"
    )
    (trials / "trials.json").write_text("")

    storage = JsonlStorageService(plants_dir=str(plants), trials_dir=str(trials))
    assert list(storage.load_plants().keys()) == ["p"]
