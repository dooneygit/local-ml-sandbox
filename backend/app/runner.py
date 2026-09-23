"""Single background worker that trains queued runs one at a time.

Epoch events are appended to an in-memory list per run, so any number of SSE
clients (and reconnects) can replay them from the start.
"""

import queue
import threading
from dataclasses import dataclass, field

from app import store
from app.ml.train import train
from app.schemas import RunConfig

TERMINAL = {"completed", "cancelled", "failed"}


@dataclass
class _Live:
    history: list[dict] = field(default_factory=list)
    cancel: threading.Event = field(default_factory=threading.Event)


class Runner:
    def __init__(self):
        self._jobs: queue.Queue[str] = queue.Queue()
        self._live: dict[str, _Live] = {}
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        # Runs left queued/running by a previous process can never finish.
        for run in store.list_runs():
            if run["status"] not in TERMINAL:
                store.update_run(run["id"], status="failed", error="Interrupted by server restart.")
        self._thread = threading.Thread(target=self._work, daemon=True)
        self._thread.start()

    def submit(self, run_id: str) -> None:
        self._live[run_id] = _Live()
        self._jobs.put(run_id)

    def cancel(self, run_id: str) -> None:
        if run_id in self._live:
            self._live[run_id].cancel.set()

    def history(self, run_id: str) -> list[dict]:
        live = self._live.get(run_id)
        return live.history if live else store.get_artifact(run_id, "history", [])

    def _work(self) -> None:
        while True:
            run_id = self._jobs.get()
            live = self._live[run_id]
            if live.cancel.is_set():
                store.update_run(run_id, status="cancelled")
                continue

            run = store.update_run(run_id, status="running")
            try:
                result = train(
                    RunConfig(**run["config"]),
                    store.load_dataset(run["config"]["dataset_id"]),
                    store.run_dir(run_id) / "model.keras",
                    on_epoch=live.history.append,
                    should_stop=live.cancel.is_set,
                )
                store.save_artifact(run_id, "history", live.history)
                store.save_artifact(run_id, "predictions", result["predictions"])
                store.update_run(
                    run_id,
                    status="cancelled" if live.cancel.is_set() else "completed",
                    task=result["task"],
                    classes=result["classes"],
                    metrics=result["metrics"],
                )
            except Exception as error:
                store.save_artifact(run_id, "history", live.history)
                store.update_run(run_id, status="failed", error=str(error))


runner = Runner()
