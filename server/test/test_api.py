import json
import os
import uuid

import pytest
from fastapi.testclient import TestClient

from api import create_api
from application_interface import ApplicationInterface
from samples_service import SamplesService
from services import SqliteStorageService


class MockApplicationLayer(ApplicationInterface):
    def __init__(self):
        self.plants = {}
        self.image_folder = "images"
        self.response_guid = None

    def set_reponse_guid(self, guid):
        self.response_guid = guid

    def segment_plant(self, image, task='leaf', data=None):
        guid = self.response_guid if self.response_guid else str(uuid.uuid4())
        self.plants[guid] = {
            "plant_id": guid,
            "status": "submitted",
            "image": f"{guid}.jpeg",
            "masks": [[[0, 0], [1, 1]]],
            "bounding_boxes": [[0.25, 0.25, 0.5, 0.5]]
        }
        return guid

    def create_plant(self, plant_id, status="submitted", image=None):
        self.plants[plant_id] = {
            "plant_id": plant_id,
            "status": status,
            "image": f"{plant_id}.jpeg" if image is None else image,
            "masks": [[[0, 0], [1, 1]]],
            "bounding_boxes": [[0.25, 0.25, 0.5, 0.5]]
        }

    def set_plant_status(self, plant_id, status):
        self.plants[plant_id]["status"] = status

    def clear_plants(self):
        self.plants = {}

    def plant_status(self, plant_id):
        if plant_id in self.plants:
            return self.plants[plant_id]["status"]
        return "plant_not_found"

    def plant_data(self, plant_id):
        response = {"plant_id": "", "status": "", "image": "", "masks": [], "bounding_boxes": []}
        if plant_id in self.plants:
            response = self.plants[plant_id]
        return response

    def get_image(self, plant_id, image_name):
        plants = {"test_guid": {"image": "0025.jpg", "segmentation": "0025_segmentation.png"}}
        try:
            file = plants[plant_id][image_name]
            with open(file, 'rb') as fs:
                data = fs.read()
            return data, "image/png" if ("png" in file) else "image/jpeg"
        except Exception:
            return None, None

    def get_plant_ids(self):
        return list(self.plants.keys())

    def get_trials(self):
        return []

    def create_trial(self, trial_data):
        return "test_trial_id"


mock_application_layer = MockApplicationLayer()


@pytest.fixture()
def client():
    app = create_api("test", application_layer=mock_application_layer)
    return TestClient(app)


def test_plant_submission_valid(client):
    # Given
    guid = str(uuid.uuid4())
    mock_application_layer.clear_plants()
    mock_application_layer.set_reponse_guid(guid)

    # When
    with open('0025.jpg', 'rb') as image:
        response = client.put("/dl/segmentation", files={"image": image})

    # Then
    assert response.status_code == 200
    assert guid == response.json()["plant_id"]


def test_plant_submission_bad_message_type(client):
    # When — JSON body is rejected by the multipart-only endpoint
    response = client.put("/dl/segmentation", json={"image": "not an image"})

    # Then
    assert response.status_code == 400


def test_plant_submission_no_image(client):
    # When — multipart body without an "image" field
    response = client.put(
        "/dl/segmentation",
        files={"not_image": ("foo.txt", b"not an image", "text/plain")},
    )

    # Then
    assert response.status_code == 400


def test_plant_status_valid(client):
    # Given
    plant_id = 'test_guid'
    mock_application_layer.clear_plants()
    mock_application_layer.create_plant(plant_id, status="test_status")

    # When
    response = client.get("/plant/status", params={"plant_id": plant_id})

    # Then
    assert response.json()["status"] == "test_status"


def test_plant_status_bad(client):
    # When — query string omits plant_id; layer returns "plant_not_found"
    response = client.get("/plant/status", params={"plant_id": "not real id"})

    # Then
    assert response.json()["status"] == "plant_not_found"


def test_plant_data_valid(client):
    # Given
    plant_id = 'test_guid'
    mock_application_layer.clear_plants()
    mock_application_layer.create_plant(plant_id, status="complete")

    # When
    response = client.get("/plant/data", params={"plant_id": plant_id})

    # Then
    assert response.json() == {
        "plant_id": plant_id,
        "status": "complete",
        "image": f"{plant_id}.jpeg",
        "masks": [[[0, 0], [1, 1]]],
        "bounding_boxes": [[0.25, 0.25, 0.5, 0.5]],
    }


def test_plant_data_bad(client):
    # Given
    mock_application_layer.clear_plants()
    mock_application_layer.create_plant('test_guid', status="complete")

    # When
    response = client.get("/plant/data", params={"plant_id": "not real id"})

    # Then
    assert response.json() == {
        "plant_id": "",
        "status": "",
        "image": "",
        "masks": [],
        "bounding_boxes": [],
    }


def test_plant_get_image_valid_image(client):
    # When
    response = client.get(
        "/plant/image",
        params={"plant_id": "test_guid", "image_name": "image"},
    )

    # Then
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/jpeg")
    with open('0025.jpg', 'rb') as f:
        assert response.content == f.read()


def test_plant_get_image_valid_segmentation(client):
    # When
    response = client.get(
        "/plant/image",
        params={"plant_id": "test_guid", "image_name": "segmentation"},
    )

    # Then
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    with open('0025_segmentation.png', 'rb') as f:
        assert response.content == f.read()


def test_plant_handles_bad_image_name_grab(client):
    # When
    response = client.get(
        "/plant/image",
        params={"plant_id": "test_guid", "image_name": "image_not_found"},
    )

    # Then
    assert response.status_code == 400


def test_plant_handles_bad_guid(client):
    # When
    response = client.get(
        "/plant/image",
        params={"plant_id": "bad_test_guid", "image_name": "image_not_found"},
    )

    # Then
    assert response.status_code == 400


def test_get_plant_ids(client):
    # Given
    plant_ids = ['1', '2', '3', '4', '5']
    mock_application_layer.clear_plants()
    for plant_id in plant_ids:
        mock_application_layer.create_plant(plant_id)

    # When
    response = client.get("/plant/ids")

    # Then
    assert response.status_code == 200
    assert response.json() == {"plant_ids": plant_ids}


def test_get_plant_id_no_ids(client):
    # Given
    mock_application_layer.clear_plants()

    # When
    response = client.get("/plant/ids")

    # Then
    assert response.status_code == 200
    assert response.json() == {"plant_ids": []}


def test_status_endpoint(client):
    # When
    response = client.get("/status")

    # Then
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_completions_disabled_when_no_chatbot(client):
    # When — chatbot is not configured in the test fixture
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gopher-eye-grape-leaf", "messages": []},
    )

    # Then — 503 with OpenAI-style error envelope
    assert response.status_code == 503
    body = response.json()
    assert body["error"]["type"] == "server_error"
    assert "chatbot is not configured" in body["error"]["message"]


def test_chat_completions_rejects_streaming(client):
    # Given — chatbot stub that should never be invoked because stream=true is rejected first
    class _StubChatbot:
        def list_models(self):
            return {"object": "list", "data": []}

        def chat_completion(self, payload):  # pragma: no cover
            raise AssertionError("chat_completion should not be called when stream=true")

    streaming_app = create_api(
        "test-stream",
        application_layer=mock_application_layer,
        chatbot=_StubChatbot(),
    )
    streaming_client = TestClient(streaming_app)

    # When
    response = streaming_client.post(
        "/v1/chat/completions",
        json={"model": "gopher-eye-grape-leaf", "messages": [], "stream": True},
    )

    # Then
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["param"] == "stream"
    assert body["error"]["type"] == "invalid_request_error"


# ---------------- /samples (mobile sync) ----------------


@pytest.fixture()
def samples_client(tmp_path):
    storage = SqliteStorageService(db_path=str(tmp_path / "samples.sqlite3"))
    samples = SamplesService(storage=storage, source_dir=str(tmp_path / "sources"))
    app = create_api(
        "test-samples",
        application_layer=mock_application_layer,
        samples_service=samples,
    )
    try:
        yield TestClient(app)
    finally:
        storage.close()


def _create_sample(client, **metadata):
    # Mobile uploads as multipart with the metadata in a JSON-encoded ``data``
    # field. Mirror that exactly so the test exercises the same code path.
    return client.post(
        "/samples",
        files={"image": ("capture.jpg", b"\xff\xd8\xff\xd9", "image/jpeg")},
        data={"data": json.dumps(metadata)},
    )


def test_create_sample_round_trips_collection_and_qr_fields(samples_client):
    # When — mobile pushes a sample with the full QR + collection metadata bundle
    response = _create_sample(
        samples_client,
        taken_at=1_700_000_000_000,
        latitude=44.97,
        longitude=-93.23,
        accuracy=4.5,
        detection_mode="wheat_fhb",
        user="alice",
        collection_id=42,
        qr_id="QR-001",
        qr_line="line-3",
        qr_rep="rep-2",
        qr_location="north-plot",
        qr_note="visible scab",
    )

    # Then — fields survive create_sample → SQLite → serialize
    assert response.status_code == 201
    sample = response.json()
    assert sample["collection_id"] == 42
    assert sample["qr_id"] == "QR-001"
    assert sample["qr_line"] == "line-3"
    assert sample["qr_rep"] == "rep-2"
    assert sample["qr_location"] == "north-plot"
    assert sample["qr_note"] == "visible scab"
    sample_id = sample["id"]

    # And the same fields come back from GET /samples/{id} and the list endpoint
    fetched = samples_client.get(f"/samples/{sample_id}").json()
    assert fetched["collection_id"] == 42
    assert fetched["qr_id"] == "QR-001"

    listed = samples_client.get("/samples").json()["samples"]
    assert any(
        s["id"] == sample_id and s["collection_id"] == 42 and s["qr_id"] == "QR-001"
        for s in listed
    )


def test_create_sample_without_collection_or_qr_defaults_to_null(samples_client):
    # When — sample with no grouping metadata (e.g. ad-hoc capture)
    response = _create_sample(
        samples_client,
        taken_at=1_700_000_000_000,
        detection_mode="grape_leaf",
        user="bob",
    )

    # Then — optional columns are returned as nulls rather than missing keys
    assert response.status_code == 201
    sample = response.json()
    for key in ("collection_id", "qr_id", "qr_line", "qr_rep", "qr_location", "qr_note"):
        assert key in sample
        assert sample[key] is None


def test_put_sample_blob_returns_serialized_sample(samples_client):
    # Given — a sample exists
    created = _create_sample(samples_client, detection_mode="wheat_fhb").json()
    sample_id = created["id"]

    # When — mobile uploads the working_image_png blob with width/height
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
    response = samples_client.put(
        f"/samples/{sample_id}/blob/working_image_png?width=1024&height=768",
        content=png,
        headers={"Content-Type": "image/png"},
    )

    # Then — server returns the updated sample (200 + JSON body, not 204) so the
    # mobile client can pick up has_working_image_png / dimensions in one trip.
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == sample_id
    assert body["has_working_image_png"] is True
    assert body["working_image_w"] == 1024
    assert body["working_image_h"] == 768
    # Blob bytes are not inlined unless include_blobs is requested.
    assert body["working_image_png"] is None
