import os
import sys

# Make the `app/` package importable for tests, mirroring how `script/server`
# runs `python app/app.py` from the repo root. Without this, tests that do
# `from application import …` only work when PYTHONPATH=app is set manually.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))
