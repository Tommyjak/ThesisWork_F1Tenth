import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from benchmarking.pipeline import run_benchmark
import preprocessing.configs.exp_yaml_parser as eyp

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
EVAL_DIR = BASE_DIR / "evaluation"
PREDICTIONS_DIR = EVAL_DIR / "predictions"
LEDGER_PATH = EVAL_DIR / "ledger.jsonl"

BASELINE_MODELS = ["yolo26n.pt"]

def collect_models():
    models = []

    for name in BASELINE_MODELS:
        weights = MODELS_DIR / name

        if not weights.is_file():
            print(f"{name} non trovato, skip")
            continue

        models.append((weights.stem, weights))

    if not RESULTS_DIR.is_dir():
        return models

    for run in sorted(RESULTS_DIR.iterdir()):
        if not run.is_dir():
            continue

        weights = run / "weights" / "best.pt"

        if not weights.is_file():
            print(f"non ho trovato i pesi di {run.name}")
            continue

        models.append((run.name, weights))

    return models

def get_extra(run_name):
    meta = eyp.get_exp_meta()

    if run_name not in meta:
        return {'arm': 'baseline', 'tag': None}

    return {'arm': meta[run_name]['arm'], 'tag': meta[run_name]['tag']}

def fmt(value):
    if value is None:
        return "-"

    return f"{value:.4f}"

def main():
    models = collect_models()

    if not models:
        raise RuntimeError("Non ho trovato i modelli")

    for run_name, weights in models:
        rows = run_benchmark(weights, run_name, PREDICTIONS_DIR, LEDGER_PATH, extra = get_extra(run_name))

        for row in rows:
            print(f"[{row['run']}] {row['benchmark']}: mAP50={fmt(row['all/mAP50'])} FAR={fmt(row['negatives/far'])}")

if __name__ == "__main__":
    main()