from pathlib import Path
import json

from benchmarking.ledger import append_row, build_row
from benchmarking.loading import load_benchmarks
from benchmarking.metrics import CONF_OP, compute_metrics
from benchmarking.prediction import predict_benchmark
from benchmarking.slices import apply_slice, build_slices

def get_dump_path(prediction_dir, run_name, benchmark_name):
    return Path(prediction_dir) / run_name / f"{benchmark_name}.jsonl"

def run_predictions(weights, run_name, prediction_dir, device = None):
    benchmarks = load_benchmarks()

    if not benchmarks:
        raise RuntimeError("Nessun benchmark trovato da benchmarks.yaml")

    dumps = []

    for benchmark in benchmarks:
        out_path = get_dump_path(prediction_dir, run_name, benchmark.name)
        dump = predict_benchmark(weights, benchmark, out_path, device)
        dumps.append(dump)

    return dumps


def read_dump(dump_path):
    records = []

    with open(dump_path, 'r') as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return records

def read_meta(dump_path):
    meta_path = Path(dump_path).with_suffix('.meta.json')

    if not meta_path.is_file():
        return None

    with open(meta_path, 'r') as f:
        return json.load(f)

def evaluate_dump(dump_path, conf_op = CONF_OP):
    records = read_dump(dump_path)

    if not records:
        raise RuntimeError(f"Dump vuoto: {dump_path}")

    specs = build_slices(records)
    slice_results = {}

    for spec in specs:
        subset = apply_slice(records, spec)
        slice_results[spec['name']] = compute_metrics(subset, conf_op)

    return slice_results

def run_benchmark(weights, run_name, prediction_dir, ledger_path, device = None, extra = None, conf_op = CONF_OP):
    benchmarks = load_benchmarks()

    if not benchmarks:
        raise RuntimeError("Nessun benchmark trovato da benchmarks.yaml")

    rows = []

    for benchmark in benchmarks:
        out_path = get_dump_path(prediction_dir, run_name, benchmark.name)
        dump = predict_benchmark(weights, benchmark, out_path, device)
        slice_results = evaluate_dump(dump, conf_op)
        meta = read_meta(dump)
        row = build_row(run_name, benchmark.name, slice_results, conf_op, meta, extra)
        append_row(ledger_path, row)
        rows.append(row)

    return rows