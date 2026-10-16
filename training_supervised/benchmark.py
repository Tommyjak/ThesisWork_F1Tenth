import re
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent))

from benchmarking.pipeline import run_benchmark
import preprocessing.configs.exp_yaml_parser as eyp

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
EVAL_DIR = BASE_DIR / "evaluation"
PREDICTIONS_DIR = EVAL_DIR / "predictions"
LEDGER_PATH = EVAL_DIR / "ledger.jsonl"

# i modelli mai allenati, uno per taglia: sono il b[j] del forward transfer, e
# analysis.py li cerca per nome via BASELINE_BY_TAG. Senza yolo26s e yolo26m le
# tabelle di continual learning dei tag "s" ed "m" avrebbero la riga FWT vuota
BASELINE_MODELS = ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt"]

# quali checkpoint valutare per ogni run allenata.
#
# best.pt e' scelto da ultralytics sull'epoca con la fitness piu' alta sulla
# validazione DI QUELLA RUN - e la validazione ha una composizione diversa per
# ogni braccio: 100% icra per ft, 75% per ftr_10, 23% per joint_w100. Per ft
# e' quindi l'epoca che massimizza l'adattamento, cioe' per costruzione quella
# di massimo oblio; per il joint e' scelto su una val al 77% di dominio vecchio.
# Confrontare best.pt fra bracci confronda criteri di selezione opposti.
#
# last.pt e' l'epoca finale, uguale per tutti (30 o 100): budget fisso, nessuna
# selezione. E' lo standard del continual learning. Si valutano entrambi, cosi'
# la differenza best-vs-last diventa un risultato esplicito - quanto dell'oblio
# di ft viene dalla scelta del checkpoint - invece di una nota a pie' di pagina.
#
# La run a last.pt prende il nome "<run>@last": run distinta nel ledger, dump
# distinto in predictions/, tag "<tag>@last" cosi' analysis.py costruisce una
# seconda tabella di continual learning parallela senza celle ambigue.
CHECKPOINTS = ["best", "last"]
LAST_SUFFIX = "@last"

# --- i checkpoint intermedi ----------------------------------------------------
#
# Con save_period: N nel blocco hyp, ultralytics scrive weights/epoch{k}.pt ogni
# N epoche. ATTENZIONE all'off-by-one: k e' l'indice 0-based del ciclo di
# training (trainer.py: `self.epoch % save_period == 0`), quindi epoch0.pt e' il
# modello DOPO LA PRIMA epoca, epoch2.pt dopo la terza, e cosi' via.
#
# Nel ledger la run si chiama "<run>@ep<E>" con E = epoche COMPLETATE = k + 1:
# e' cio' che un lettore si aspetta, e coincide con la colonna `epoch` di
# results.csv, da cui si legge il tempo cumulativo di training a quel punto.
# Con questi due campi - epoch e train_s - analysis.py ricostruisce la
# traiettoria di ogni run sui benchmark: mAP e FAR contro il tempo speso.
#
# Le righe @ep portano il tag "<tag>@ep": cl_metrics le ignora (cerca una cella
# per (arm, tag) e queste non collidono con "main" ne' con "main@last"), quindi
# le tabelle finali non cambiano.
EPOCH_SUFFIX = "@ep"
EPOCH_FILE = re.compile(r"^epoch(\d+)\.pt$")

_RESULTS_CSV = {}

# la tabella per-epoca di ultralytics, letta una volta per run. Colonne con
# spazi ai bordi ('   epoch') nelle versioni vecchie: si strippano
def results_csv(run_dir):
    run_dir = Path(run_dir)

    if run_dir not in _RESULTS_CSV:
        csv = run_dir / "results.csv"
        table = None

        if csv.is_file():
            try:
                table = pd.read_csv(csv)
                table.columns = [c.strip() for c in table.columns]
            except Exception:
                table = None

        _RESULTS_CSV[run_dir] = table

    return _RESULTS_CSV[run_dir]

# secondi cumulativi di training dopo `epochs` epoche, da results.csv
def train_seconds(run_dir, epochs):
    table = results_csv(run_dir)

    if table is None or 'time' not in table.columns or 'epoch' not in table.columns:
        return None

    row = table[table['epoch'] == epochs]

    return float(row['time'].iloc[0]) if not row.empty else None

# l'epoca di best.pt non e' nel checkpoint (viene azzerata a fine training):
# si ricava da results.csv come la riga con la fitness piu' alta, con la stessa
# formula di ultralytics, 0.1 * mAP50 + 0.9 * mAP50-95 sulla validazione
def best_epoch(run_dir):
    table = results_csv(run_dir)

    if table is None:
        return None

    m50 = next((c for c in table.columns if c.endswith('mAP50(B)')), None)
    m95 = next((c for c in table.columns if c.endswith('mAP50-95(B)')), None)

    if m50 is None or m95 is None:
        return None

    fitness = 0.1 * table[m50] + 0.9 * table[m95]

    return int(table['epoch'].iloc[int(fitness.values.argmax())])

def last_epoch(run_dir):
    table = results_csv(run_dir)

    return int(table['epoch'].iloc[-1]) if table is not None and not table.empty else None

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

    # solo le run definite nello yaml degli esperimenti ATTIVO (EXP_YAML). Cosi'
    # con EXP_YAML=tmp_experiments.yaml e BM_YAML=tmp_benchmarks.yaml si valutano
    # le sole run sequenziali sui 5 holdout, e le 42 della campagna principale -
    # che hanno visto in training le immagini di f1tenth_holdout e
    # squeaky_holdout - non finiscono nel ledger con numeri contaminati.
    # Il ledger fa upsert per (run, benchmark): le due passate si sommano
    known = set(eyp.get_exp_meta())
    skipped = []

    for run in sorted(RESULTS_DIR.iterdir()):
        if not run.is_dir():
            continue

        if run.name not in known:
            skipped.append(run.name)
            continue

        for checkpoint in CHECKPOINTS:
            weights = run / "weights" / f"{checkpoint}.pt"

            if not weights.is_file():
                print(f"non ho trovato {checkpoint}.pt di {run.name}")
                continue

            run_name = run.name if checkpoint == "best" else run.name + LAST_SUFFIX
            models.append((run_name, weights))

        # gli intermedi, in ordine di epoca. Nessuno = run senza save_period,
        # va bene lo stesso: si valutano solo best e last
        epochs = []

        for weights in (run / "weights").glob("epoch*.pt"):
            match = EPOCH_FILE.match(weights.name)

            if match:
                epochs.append((int(match.group(1)) + 1, weights))   # k -> epoche completate

        for done, weights in sorted(epochs):
            models.append((f"{run.name}{EPOCH_SUFFIX}{done}", weights))

    if skipped:
        print(f"{len(skipped)} run in results/ non definite in {eyp.EXP_YAML_PATH.name}, non valutate: "
              + ", ".join(skipped[:6]) + (" ..." if len(skipped) > 6 else ""))

    return models

# il nome dell'esperimento in experiments.yaml, il tipo di checkpoint e - per
# gli intermedi - le epoche completate, dal nome di run
def split_run_name(run_name):
    if run_name.endswith(LAST_SUFFIX):
        return run_name[:-len(LAST_SUFFIX)], "last", None

    stem, sep, done = run_name.rpartition(EPOCH_SUFFIX)

    if sep and done.isdigit():
        return stem, "epoch", int(done)

    return run_name, "best", None

def get_extra(run_name):
    meta = eyp.get_exp_meta()
    exp_name, checkpoint, done = split_run_name(run_name)

    if exp_name not in meta:
        return {'arm': 'baseline', 'tag': None, 'checkpoint': 'pretrained',
                'epoch': None, 'train_s': None}

    tag = meta[exp_name]['tag']
    run_dir = RESULTS_DIR / exp_name

    # il tag marcato tiene le righe @last e @ep in celle (arm, tag) separate da
    # quelle best: stessa tabella di continual learning, copie parallele, e gli
    # intermedi fuori da tutte
    if checkpoint == "last":
        done = last_epoch(run_dir)
        tag = tag + LAST_SUFFIX if tag is not None else None
    elif checkpoint == "epoch":
        tag = tag + EPOCH_SUFFIX if tag is not None else None
    else:
        done = best_epoch(run_dir)

    return {
        'arm': meta[exp_name]['arm'],
        'tag': tag,
        'checkpoint': checkpoint,
        'epoch': done,
        'train_s': train_seconds(run_dir, done) if done is not None else None,
    }

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
            when = f" ep={row['epoch']} t={row['train_s']:.0f}s" if row.get('train_s') is not None else ""
            print(f"[{row['run']}]{when} {row['benchmark']}: mAP50={fmt(row['all/mAP50'])} FAR={fmt(row['negatives/far'])}")

if __name__ == "__main__":
    main()