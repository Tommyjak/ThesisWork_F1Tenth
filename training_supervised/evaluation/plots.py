import sys
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))

from benchmarking.metrics import IOU_OP, collect_entries, count_gt

EVAL_DIR = Path(__file__).resolve().parent
LEDGER_PATH = EVAL_DIR / "ledger.jsonl"
PREDICTIONS_DIR = EVAL_DIR / "predictions"
PLOTS_DIR = EVAL_DIR / "plots"

ARM_COLORS = {
    'baseline': '#999999',
    'base': '#4c72b0',
    'joint': '#55a868',
    'fine_tune': '#c44e52',
    'replay': '#dd8452',
}

TAG_MARKERS = {
    'full': 'o',
    'balanced': 's',
    'few': '^',
}

TAG_STYLES = {
    'full': '-',
    'balanced': '--',
    'few': ':',
}

TAG_ORDER = ['full', 'balanced', 'few']

def load_rows(ledger_path = LEDGER_PATH):
    rows = []

    with open(ledger_path, 'r') as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            rows.append(json.loads(line))

    return rows

def select(rows, benchmark):
    selected = []

    for row in rows:
        if row['benchmark'] == benchmark:
            selected.append(row)

    return selected

def get_arms(rows):
    arms = []

    for row in rows:
        arm = row.get('arm')

        if arm is not None and arm not in arms and arm != 'baseline':
            arms.append(arm)

    return arms

def get_color(arm):
    return ARM_COLORS.get(arm, '#333333')

def save(fig, name):
    PLOTS_DIR.mkdir(parents = True, exist_ok = True)
    path = PLOTS_DIR / name

    fig.savefig(path, dpi = 150, bbox_inches = 'tight')
    plt.close(fig)

    print(f"  {path}")

    return path

# --- 1. barre per braccio x composizione, con intervalli di confidenza -------

def plot_bars(rows, benchmark, column, title, name, lo_column = None, hi_column = None):
    rows = select(rows, benchmark)
    arms = get_arms(rows)

    fig, ax = plt.subplots(figsize = (7, 4.5))

    width = 0.8 / max(len(arms), 1)

    for i, arm in enumerate(arms):
        values = []
        errors_lo = []
        errors_hi = []
        positions = []

        for j, tag in enumerate(TAG_ORDER):
            value = None

            for row in rows:
                if row.get('arm') == arm and row.get('tag') == tag:
                    value = row.get(column)

                    if lo_column is not None:
                        lo = row.get(lo_column)
                        hi = row.get(hi_column)

            if value is None:
                continue

            positions.append(j + (i - (len(arms) - 1) / 2) * width)
            values.append(value)

            if lo_column is not None and lo is not None and hi is not None:
                errors_lo.append(value - lo)
                errors_hi.append(hi - value)
            else:
                errors_lo.append(0.0)
                errors_hi.append(0.0)

        bars = ax.bar(positions, values, width = width * 0.9, label = arm,
                      color = get_color(arm))

        # le barre d'errore solo dove un intervallo esiste davvero
        if lo_column is not None:
            ax.errorbar(positions, values, yerr = [errors_lo, errors_hi],
                        fmt = 'none', capsize = 3, ecolor = '#444444')

    # la baseline non allenata come linea di riferimento
    for row in rows:
        if row.get('arm') == 'baseline' and row.get(column) is not None:
            ax.axhline(row[column], color = '#999999', linestyle = '--', linewidth = 1)
            ax.text(len(TAG_ORDER) - 0.5, row[column], f" {row['run']}",
                    va = 'bottom', ha = 'right', fontsize = 8, color = '#666666')

    ax.set_xticks(range(len(TAG_ORDER)))
    ax.set_xticklabels(TAG_ORDER)
    ax.set_xlabel("composizione dei negativi")
    ax.set_ylabel(column)
    ax.set_title(f"{title}  —  {benchmark}")
    ax.legend(title = "strategia", frameon = False)
    ax.grid(axis = 'y', alpha = 0.3)

    return save(fig, name)

# --- 2. il compromesso: quanto vede contro quanto inventa --------------------

def plot_tradeoff(rows, benchmark, name = "tradeoff.png"):
    rows = select(rows, benchmark)

    fig, ax = plt.subplots(figsize = (6.5, 5))

    for row in rows:
        far = row.get('negatives/far')
        recall = row.get('positives/recall')

        if far is None or recall is None:
            continue

        ax.scatter(far, recall,
                   s = 90,
                   color = get_color(row.get('arm')),
                   marker = TAG_MARKERS.get(row.get('tag'), 'o'),
                   edgecolor = 'white', zorder = 3)

        ax.annotate(row['run'], (far, recall), textcoords = 'offset points',
                    xytext = (7, 4), fontsize = 8)

    ax.set_xscale('log')
    ax.set_xlabel("FAR  —  frazione di negativi con un falso allarme  (meglio a sinistra)")
    ax.set_ylabel("recall sui positivi  (meglio in alto)")
    ax.set_title(f"Sicurezza contro copertura  —  {benchmark}")
    ax.grid(alpha = 0.3)

    return save(fig, name)

# --- 3. curve precision-recall, dai dump ------------------------------------

def read_dump(dump_path):
    records = []

    with open(dump_path, 'r') as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return records

def pr_curve(records, iou_thr = IOU_OP):
    n_gt = count_gt(records)

    if n_gt == 0:
        return [], []

    entries = collect_entries(records, iou_thr)

    recalls = [0.0]
    precisions = [1.0]

    tp = 0
    fp = 0

    for _, is_tp in entries:
        if is_tp:
            tp += 1
        else:
            fp += 1

        recalls.append(tp / n_gt)
        precisions.append(tp / (tp + fp))

    return recalls, precisions

def plot_pr(rows, benchmark, name = "pr_curves.png"):
    rows = select(rows, benchmark)

    fig, ax = plt.subplots(figsize = (6.5, 5))

    for row in rows:
        dump_path = PREDICTIONS_DIR / row['run'] / f"{benchmark}.jsonl"

        if not dump_path.is_file():
            print(f"  dump mancante, salto: {dump_path}")
            continue

        recalls, precisions = pr_curve(read_dump(dump_path))

        if not recalls:
            continue

        style = TAG_STYLES.get(row.get('tag'), '-.')

        ax.plot(recalls, precisions, linestyle = style, linewidth = 1.6,
                color = get_color(row.get('arm')),
                label = f"{row['run']} (AP={row.get('all/mAP50'):.3f})")

    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Curve precision-recall a IoU 0.5  —  {benchmark}")
    ax.legend(fontsize = 8, frameon = False)
    ax.grid(alpha = 0.3)

    return save(fig, name)

# --- 4. profilo per fetta: dove ogni modello cede ----------------------------

def get_slice_columns(rows, metric = 'mAP50'):
    columns = []

    for row in rows:
        for key in row:
            if key.endswith('/' + metric) and key not in columns:
                columns.append(key)

    return columns

def plot_slices(rows, benchmark, metric = 'mAP50', name = "slices.png"):
    rows = select(rows, benchmark)
    columns = get_slice_columns(rows, metric)

    grid = []
    labels = []

    for row in rows:
        line = []

        for column in columns:
            value = row.get(column)
            line.append(float('nan') if value is None else value)

        grid.append(line)
        labels.append(row['run'])

    fig, ax = plt.subplots(figsize = (1.1 * len(columns) + 3, 0.5 * len(labels) + 2))

    image = ax.imshow(grid, cmap = 'viridis', aspect = 'auto', vmin = 0, vmax = 1)

    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([c.split('/')[0] for c in columns], rotation = 45, ha = 'right')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(columns)):
            value = grid[i][j]
            text = "-" if value != value else f"{value:.2f}"

            ax.text(j, i, text, ha = 'center', va = 'center', fontsize = 8,
                    color = 'white' if value == value and value < 0.6 else 'black')

    ax.set_title(f"{metric} per fetta  —  {benchmark}")
    fig.colorbar(image, ax = ax, shrink = 0.7)

    return save(fig, name)

# --- 5. stabilita' contro plasticita': la figura del continual learning ------

def plot_stability(rows, adaptation, forgetting, name = "stability_plasticity.png"):
    by_run = {}

    for row in rows:
        by_run.setdefault(row['run'], {})[row['benchmark']] = row

    fig, ax = plt.subplots(figsize = (6.5, 5))
    drawn = 0

    for run in by_run:
        if adaptation not in by_run[run] or forgetting not in by_run[run]:
            continue

        x = by_run[run][adaptation].get('all/mAP50')
        y = by_run[run][forgetting].get('all/mAP50')

        if x is None or y is None:
            continue

        arm = by_run[run][adaptation].get('arm')
        tag = by_run[run][adaptation].get('tag')

        ax.scatter(x, y, s = 90, color = get_color(arm),
                   marker = TAG_MARKERS.get(tag, 'o'), edgecolor = 'white', zorder = 3)
        ax.annotate(run, (x, y), textcoords = 'offset points', xytext = (7, 4), fontsize = 8)
        drawn += 1

    if drawn == 0:
        plt.close(fig)
        print(f"  salto stability/plasticity: servono entrambi i benchmark ({adaptation}, {forgetting})")
        return None

    ax.set_xlabel(f"adattamento  —  mAP50 su {adaptation}")
    ax.set_ylabel(f"ritenzione  —  mAP50 su {forgetting}")
    ax.set_title("Stabilita' contro plasticita'  (l'angolo in alto a destra vince)")
    ax.grid(alpha = 0.3)

    return save(fig, name)

def main():
    rows = load_rows()

    benchmarks = []

    for row in rows:
        if row['benchmark'] not in benchmarks:
            benchmarks.append(row['benchmark'])

    print("figure generate:")

    for benchmark in benchmarks:
        plot_bars(rows, benchmark, 'all/mAP50', "Accuratezza", f"bars_map50_{benchmark}.png")
        plot_bars(rows, benchmark, 'negatives/far', "Falsi allarmi sui negativi",
                  f"bars_far_{benchmark}.png", 'negatives/far_lo', 'negatives/far_hi')
        plot_tradeoff(rows, benchmark, f"tradeoff_{benchmark}.png")
        plot_pr(rows, benchmark, f"pr_curves_{benchmark}.png")
        plot_slices(rows, benchmark, 'mAP50', f"slices_{benchmark}.png")

    plot_stability(rows, 'icra_holdout', 'forgetting_holdout')

if __name__ == "__main__":
    main()
