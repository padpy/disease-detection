import pytest

from services import JsonlStorageService, SqliteStorageService, migrate


@pytest.fixture
def jsonl_with_data(tmp_path):
    plants = tmp_path / "plants"
    trials = tmp_path / "trials"
    svc = JsonlStorageService(plants_dir=str(plants), trials_dir=str(trials))
    svc.save_plant({
        "plant_id": "p1",
        "status": "complete",
        "image": "p1.jpeg",
        "bounding_boxes": [[0.1, 0.1, 0.2, 0.2]],
        "masks": [[[0.0, 0.0], [1.0, 1.0]]],
        "labels": ["Healthy-Leaf"],
        "trial_id": "t1",
    })
    svc.save_plant({"plant_id": "p2", "status": "pending", "image": "p2.jpeg"})
    svc.save_trial({"trial_id": "t1", "trial_name": "trial-one"})
    return svc


def test_migrate_jsonl_to_sqlite(tmp_path, jsonl_with_data):
    sqlite_svc = SqliteStorageService(db_path=str(tmp_path / "out.sqlite3"))
    try:
        plants_n, trials_n = migrate(jsonl_with_data, sqlite_svc)

        assert plants_n == 2
        assert trials_n == 1
        assert set(sqlite_svc.list_plant_ids()) == {"p1", "p2"}
        assert sqlite_svc.get_plant("p1")["labels"] == ["Healthy-Leaf"]
        assert sqlite_svc.get_plant("p1")["bounding_boxes"] == [[0.1, 0.1, 0.2, 0.2]]
        assert sqlite_svc.load_trials()["t1"]["trial_name"] == "trial-one"
    finally:
        sqlite_svc.close()


def test_migrate_is_idempotent(tmp_path, jsonl_with_data):
    sqlite_svc = SqliteStorageService(db_path=str(tmp_path / "out.sqlite3"))
    try:
        migrate(jsonl_with_data, sqlite_svc)
        migrate(jsonl_with_data, sqlite_svc)  # second run should not duplicate

        assert len(sqlite_svc.list_plant_ids()) == 2
        assert len(sqlite_svc.load_trials()) == 1
    finally:
        sqlite_svc.close()
