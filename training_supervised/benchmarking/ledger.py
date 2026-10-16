import json
from datetime import datetime
from pathlib import Path

def flatten_metrics(slice_name, metrics):
    flat = {}

    for key in metrics:
        flat[f"{slice_name}/{key}"] = metrics[key]

    return flat

def build_row(run_name, benchmark_name, slice_results, conf_op, meta = None, extra = None):
    row = {
        'run': run_name,
        'benchmark': benchmark_name,
        'timestamp': datetime.now().isoformat(timespec = 'seconds'),
        'conf_op': conf_op,
    }

    if meta is not None:
        row['weights'] = meta.get('weights')
        row['weights_sha'] = meta.get('weights_sha')
        row['n_images'] = meta.get('n_images')

        # macchina e tempi dell'inferenza, dal meta del dump. Colonne piatte,
        # cosi' analysis.py le legge come qualunque altra
        for key in ('host', 'device', 'torch',
                    'infer_ms_per_img', 'wall_ms_per_img', 'wall_s'):
            row[key] = meta.get(key)

    if extra is not None:
        for key in extra:
            row[key] = extra[key]

    for slice_name in slice_results:
        row.update(flatten_metrics(slice_name, slice_results[slice_name]))

    return row

def read_rows(ledger_path):
    ledger_path = Path(ledger_path)

    rows = []

    if not ledger_path.is_file():
        return rows

    with open(ledger_path, 'r') as f:
        for line in f:
            line = line.strip()

            if not line:
                continue
            rows.append(json.loads(line))

    return rows

def write_rows(ledger_path, rows):
    ledger_path = Path(ledger_path)
    ledger_path.parent.mkdir(parents = True, exist_ok = True)

    with open(ledger_path, 'w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')

def drop_row(rows, run_name, benchmark_name):
    kept = []

    for row in rows:
        if row.get('run') == run_name and row.get('benchmark') == benchmark_name:
            continue

        kept.append(row)

    return kept

def append_row(ledger_path, row):
    rows = read_rows(ledger_path)
    rows = drop_row(rows, row['run'], row['benchmark'])
    rows.append(row)

    write_rows(ledger_path, rows)

    return ledger_path