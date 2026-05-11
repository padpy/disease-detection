import os
import platform

import uvicorn

from api import create_api
from application import Application
from chatbot import build_chatbot_from_env
from samples_service import SamplesService
from services import JsonlStorageService, SqliteStorageService, migrate

# from scratch import firebase_login

DB_PATH = os.environ.get("GOPHER_EYE_DB", "data/gopher_eye.sqlite3")
LEGACY_PLANTS_DIR = os.environ.get("GOPHER_EYE_LEGACY_PLANTS", "plants")
LEGACY_TRIALS_DIR = os.environ.get("GOPHER_EYE_LEGACY_TRIALS", "trials")


def _build_storage() -> SqliteStorageService:
    """Open the SQLite store. If it didn't exist and there's a legacy JSONL
    store on disk, copy records over so existing deployments upgrade in place.
    """
    fresh_db = not os.path.exists(DB_PATH)
    storage = SqliteStorageService(db_path=DB_PATH)

    legacy_plants = os.path.join(LEGACY_PLANTS_DIR, "plants.json")
    legacy_trials = os.path.join(LEGACY_TRIALS_DIR, "trials.json")
    if fresh_db and (os.path.exists(legacy_plants) or os.path.exists(legacy_trials)):
        legacy = JsonlStorageService(
            plants_dir=LEGACY_PLANTS_DIR, trials_dir=LEGACY_TRIALS_DIR
        )
        plants_migrated, trials_migrated = migrate(legacy, storage)
        print(
            f"Migrated {plants_migrated} plants and {trials_migrated} trials "
            f"from JSONL into {DB_PATH}"
        )

    return storage


if __name__ == '__main__':
    # Initialize the firebase application
    # cred = Credentials.Certificate('creds/firebase-cred.json')
    # firebase_login.initialize_app(cred)

    chatbot = build_chatbot_from_env()
    if chatbot is None:
        print("Chatbot disabled: set GOPHER_EYE_LLM_CKPT (and GOPHER_EYE_LLM_DIR) to enable /v1/chat/completions")
    else:
        print(f"Chatbot enabled: ckpt={chatbot.ckpt_dir} (lazy load on first request)")

    storage = _build_storage()
    samples = SamplesService(
        storage=storage,
        source_dir=os.environ.get("GOPHER_EYE_SAMPLE_SOURCES", "data/sample_sources"),
    )
    app = create_api(
        __name__,
        Application(storage_service=storage),
        chatbot=chatbot,
        samples_service=samples,
    )
    port = 5555 if platform.system() == 'Darwin' else 5000
    uvicorn.run(app, host="0.0.0.0", port=port)
