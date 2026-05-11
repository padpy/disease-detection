import os
import sqlite3

import pytest

from services import SqliteStorageService


@pytest.fixture
def storage(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    svc = SqliteStorageService(db_path=str(db_path))
    yield svc
    svc.close()


def test_init_creates_db_file(tmp_path):
    db_path = tmp_path / "nested" / "test.sqlite3"
    svc = SqliteStorageService(db_path=str(db_path))
    try:
        assert os.path.exists(db_path)
    finally:
        svc.close()


def test_save_and_get_plant(storage):
    storage.save_plant({
        "plant_id": "abc",
        "status": "pending",
        "image": "abc.jpeg",
        "bounding_boxes": [[0.1, 0.2, 0.3, 0.4]],
        "masks": [[[0.0, 0.0], [0.5, 0.5]]],
        "labels": ["Powdery-Leaf"],
        "trial_id": "t1",
        "datetime": "2026-05-06T10:00:00Z",
        "plot_label_name": "row-3",
        "plot_id": "plot-7",
        "plot_location": {"lat": 44.97, "lng": -93.23},
        "user": "alice",
    })

    plant = storage.get_plant("abc")
    assert plant["status"] == "pending"
    assert plant["bounding_boxes"] == [[0.1, 0.2, 0.3, 0.4]]
    assert plant["masks"] == [[[0.0, 0.0], [0.5, 0.5]]]
    assert plant["labels"] == ["Powdery-Leaf"]
    assert plant["plot_location"] == {"lat": 44.97, "lng": -93.23}
    assert plant["user"] == "alice"


def test_save_plant_upserts(storage):
    storage.save_plant({
        "plant_id": "abc",
        "status": "pending",
        "image": "abc.jpeg",
    })
    storage.save_plant({
        "plant_id": "abc",
        "status": "complete",
        "image": "abc.jpeg",
        "labels": ["Healthy-Leaf"],
    })

    plant = storage.get_plant("abc")
    assert plant["status"] == "complete"
    assert plant["labels"] == ["Healthy-Leaf"]
    # Only one row exists
    assert len(storage.list_plant_ids()) == 1


def test_get_plant_returns_none_for_missing(storage):
    assert storage.get_plant("missing") is None


def test_list_plant_ids_preserves_insertion_order(storage):
    for pid in ["alpha", "bravo", "charlie"]:
        storage.save_plant({"plant_id": pid, "status": "pending", "image": f"{pid}.jpeg"})
    assert storage.list_plant_ids() == ["alpha", "bravo", "charlie"]


def test_load_plants_returns_all_records(storage):
    for pid in ["a", "b"]:
        storage.save_plant({"plant_id": pid, "status": "complete", "image": f"{pid}.jpeg"})
    plants = storage.load_plants()
    assert set(plants.keys()) == {"a", "b"}
    assert all(p["status"] == "complete" for p in plants.values())


def test_save_and_load_trial(storage):
    storage.save_trial({
        "trial_id": "t1",
        "trial_name": "North Field",
        "description": "rust resistance",
        "user": "alice",
    })
    trials = storage.load_trials()
    assert trials["t1"]["trial_name"] == "North Field"
    assert trials["t1"]["user"] == "alice"


def test_persists_across_connection(tmp_path):
    db_path = str(tmp_path / "persist.sqlite3")
    svc = SqliteStorageService(db_path=db_path)
    svc.save_plant({
        "plant_id": "abc",
        "status": "complete",
        "image": "abc.jpeg",
        "masks": [[[0, 0]]],
    })
    svc.close()

    svc2 = SqliteStorageService(db_path=db_path)
    try:
        plant = svc2.get_plant("abc")
        assert plant is not None
        assert plant["masks"] == [[[0, 0]]]
    finally:
        svc2.close()


def test_concurrent_writes_serialize(storage):
    """Background segmentation runs on a worker thread; the storage lock must
    keep concurrent saves from corrupting state."""
    import threading

    def worker(start):
        for i in range(start, start + 25):
            storage.save_plant({
                "plant_id": f"p{i}",
                "status": "complete",
                "image": f"p{i}.jpeg",
            })

    threads = [threading.Thread(target=worker, args=(i * 25,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(storage.list_plant_ids()) == 100


def test_plot_location_string_value_round_trips(storage):
    storage.save_plant({
        "plant_id": "abc",
        "status": "pending",
        "image": "abc.jpeg",
        "plot_location": "greenhouse-2",
    })
    plant = storage.get_plant("abc")
    assert plant["plot_location"] == "greenhouse-2"
