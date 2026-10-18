"""Local MLflow record for one training invocation.

metrics.jsonl and the ppo_*.pt files stay where they are. This copy keeps each
invocation separate, including which checkpoint it started from. The store is a
SQLite file under ``logs/mlflow``.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


class LocalTracker:
    """One MLflow run under ``logs/mlflow``. A missing install leaves training unchanged."""

    def __init__(self) -> None:
        self._mlflow = None
        self.active = False

    def start(self, log_dir: Path, info: dict) -> None:
        try:
            import mlflow
        except ImportError:
            print(
                "MLflow が無いので、ローカル記録はスキップします。metrics.jsonl はそのまま残します。",
                flush=True,
            )
            return
        store = _store_dir(log_dir)
        store.mkdir(parents=True, exist_ok=True)
        (store / "artifacts").mkdir(exist_ok=True)
        db = store / "mlflow.db"
        mlflow.set_tracking_uri("sqlite:///" + db.resolve().as_posix())
        artifact_root = (store / "artifacts").resolve().as_uri()
        if mlflow.get_experiment_by_name("pick-and-carry") is None:
            mlflow.create_experiment("pick-and-carry", artifact_location=artifact_root)
        mlflow.set_experiment("pick-and-carry")
        stage = str(info.get("curriculum", "carry"))
        level = str(info.get("catch_level", "static"))
        algo = str(info.get("algorithm", "ppo")).lower()
        run_name = f"{stage}-{level}" if stage == "catch" else stage
        # BC / GAIL share the walk log dir with PPO; prefix so runs are easy to filter.
        if algo in ("bc", "gail"):
            easy = bool(info.get("easy_walk", False))
            run_name = f"{algo}-{run_name}{'-easy' if easy and stage == 'walk' else ''}"
        mlflow.start_run(run_name=run_name)
        self._mlflow = mlflow
        self.active = True
        params = {
            "algorithm": info.get("algorithm", "ppo"),
            "curriculum": stage,
            "catch_level": level,
            "arm": info.get("arm", "z1"),
            "obs_dim": info.get("obs_dim"),
            "act_dim": info.get("act_dim"),
            "num_envs": info.get("num_envs"),
            "num_steps": info.get("num_steps"),
            "iterations_requested": info.get("iterations_requested"),
            "start_iteration": info.get("start_iteration"),
            "learning_rate": info.get("learning_rate"),
            "seed": info.get("seed"),
            "terrain": info.get("terrain"),
            "terrain_steps": info.get("terrain_steps"),
            "terrain_amplitude_m": info.get("terrain_amplitude_m"),
            "control_split": info.get("control_split"),
            "wrist_camera": info.get("wrist_camera"),
            "easy_walk": info.get("easy_walk"),
            "bc_epochs": info.get("bc_epochs"),
            "epochs": info.get("epochs"),
            "gail_coef": info.get("gail_coef"),
            "env_coef": info.get("env_coef"),
            "disc_epochs": info.get("disc_epochs"),
            "demo_seconds": info.get("demo_seconds"),
            "log_dir": str(log_dir),
        }
        params = {key: value for key, value in params.items() if value is not None}
        self._call(lambda: mlflow.log_params({key: "" if value is None else str(value) for key, value in params.items()}))
        resumed = info.get("resumed_from")
        if resumed:
            self._call(lambda: mlflow.set_tag("resumed_from", str(resumed)))
        self._call(lambda: mlflow.set_tag("algorithm", algo))
        print(f"MLflow に残します: {store}", flush=True)
        print(
            r"見るときは .\.venv\Scripts\mlflow ui --backend-store-uri sqlite:///logs/mlflow/mlflow.db --workers 1",
            flush=True,
        )

    def log_row(self, row: dict) -> None:
        if not self.active:
            return
        step = int(row["iteration"])
        metrics = {key: float(value) for key, value in row.items() if key != "iteration"}
        self._call(lambda: self._mlflow.log_metrics(metrics, step=step))

    def log_checkpoint(self, path: Path) -> None:
        if not self.active or not path.is_file():
            return
        self._call(lambda: self._mlflow.log_artifact(str(path), artifact_path="checkpoints"))

    def close(self) -> None:
        if not self.active:
            return
        mlflow = self._mlflow
        self.active = False
        try:
            mlflow.end_run()
        except Exception as exc:
            print(f"MLflow の記録を閉じられませんでした: {exc}", flush=True)

    def _call(self, action) -> None:
        try:
            action()
        except Exception as exc:
            print(f"MLflow の記録を止めます: {exc}", flush=True)
            self.close()


def release_store() -> None:
    """Close the SQLite file. Tests need this before deleting a temporary directory."""
    try:
        from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
    except ImportError:
        return
    for engine in list(SqlAlchemyStore._engine_map.values()):
        engine.dispose()
    SqlAlchemyStore._engine_map.clear()


def _store_dir(log_dir: Path) -> Path:
    """One store next to the stage folders, so every stage shares ``logs/mlflow``."""
    parent = log_dir.parent
    if parent.name == "logs":
        return parent / "mlflow"
    return Path("logs") / "mlflow"
