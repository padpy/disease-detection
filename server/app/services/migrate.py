"""One-shot migration helpers for moving data between storage backends.

Typical use: import the function, hand it a source and a destination, call it
once at startup or from a script. The operation is idempotent because both
``save_plant`` and ``save_trial`` upsert by primary key.
"""

from .storage_service import StorageService


def migrate(source: StorageService, destination: StorageService) -> tuple[int, int]:
    """Copy every plant and trial from ``source`` into ``destination``.

    Returns ``(plant_count, trial_count)``. Existing records in destination are
    overwritten on key collision.
    """
    plants = source.load_plants()
    for plant in plants.values():
        destination.save_plant(plant)

    trials = source.load_trials()
    for trial in trials.values():
        destination.save_trial(trial)

    return len(plants), len(trials)
