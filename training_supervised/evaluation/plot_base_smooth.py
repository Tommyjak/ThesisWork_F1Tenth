"""
plot_base_smooth.py
-------------------
Genera i grafici mAP50-95 e FAR per i modelli base, con le curve lisciate
tramite media mobile. Un file per ogni modello base, due pannelli affiancati
(mAP a sinistra, FAR a destra) con i tre benchmark come curve.

Uso:
    python plot_base_smooth.py
"""

import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt


# --- convenzioni (coerenti con lo script principale) -------------------------
COLONNA_MAP = 'all/mAP50_95'
COLONNA_FAR = 'negatives/far'
NOMI_METRICA = {'map': 'mAP50-95', 'far': 'FAR'}
SCALA = 100.0
LIMITE_Y = {'map': (0, 100), 'far': (0, 20)}

# i tipi dei modelli base da plottare (senza suffisso seed)
TIPI_BASE = ['base_90_10', 'base_70_30', 'base_50_50', 'base_hard', 'joint_w10', 'ft', 'ftr_10']


def formatta_asse(ax, nome_metrica, chiave):
    ax.set_ylim(*LIMITE_Y[chiave])
    ax.set_ylabel(f'{nome_metrica} (%)')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='y', which='both', left=True, right=True,
                   labelleft=True, labelright=True)


def moving_average(valori, finestra=5):
    """
    Media mobile centrata su 'finestra' epoche. Ai bordi la finestra si riduce
    ai punti disponibili, cosi' la curva liscia parte e finisce agli stessi
    estremi di quella grezza.
    """
    n = len(valori)
    if n < 2 or finestra <= 1:
        return list(valori)

    meta = finestra // 2
    lisci = []
    for i in range(n):
        lo = max(0, i - meta)
        hi = min(n, i + meta + 1)
        lisci.append(sum(valori[lo:hi]) / (hi - lo))
    return lisci


def trova_run_rappresentative(dati_plot):
    """
    Per ogni tipo (ignorando il seed) sceglie la run piu' vicina alla media dei
    seed. Versione ridotta, sufficiente per i modelli base.
    """
    import numpy as np

    tipi_run = {}
    for run_base in dati_plot.keys():
        tipo = re.sub(r'_s\d+$', '', run_base)
        tipi_run.setdefault(tipo, []).append(run_base)

    run_scelte = []
    for tipo, runs in tipi_run.items():
        if len(runs) == 1:
            run_scelte.append(runs[0])
            continue

        distanze = {r: 0.0 for r in runs}
        conteggi = {r: 0 for r in runs}
        benchmarks_comuni = set.intersection(*[set(dati_plot[r].keys()) for r in runs])
        for b in benchmarks_comuni:
            epoche_comuni = set.intersection(*[set(dati_plot[r][b]['epochs']) for r in runs])
            for epoca in epoche_comuni:
                maps = [dati_plot[r][b]['map'][dati_plot[r][b]['epochs'].index(epoca)] for r in runs]
                media = np.mean(maps)
                for r in runs:
                    idx = dati_plot[r][b]['epochs'].index(epoca)
                    distanze[r] += abs(dati_plot[r][b]['map'][idx] - media)
                    conteggi[r] += 1

        migliore = min(runs, key=lambda r: distanze[r] / max(1, conteggi[r]))
        run_scelte.append(migliore)

    return run_scelte


def carica_ledger(percorso):
    dati_plot = {}
    with open(percorso, 'r') as f:
        for linea in f:
            if not linea.strip():
                continue
            row = json.loads(linea)

            epoca = row.get('epoch')
            if epoca is None:
                continue

            run_base = row.get('run', 'sconosciuta').split('@')[0]
            benchmark = row.get('benchmark', 'sconosciuto')
            mAP = row.get(COLONNA_MAP)
            far = row.get(COLONNA_FAR)
            if mAP is None or far is None:
                continue

            d = dati_plot.setdefault(run_base, {}).setdefault(
                benchmark, {'epochs': [], 'map': [], 'far': []})
            d['epochs'].append(epoca)
            d['map'].append(mAP * SCALA)
            d['far'].append(far * SCALA)
    return dati_plot


def plot_base_smoothed(dati_plot, run_da_plottare, cartella, finestre=None):
    """
    Per ogni modello base: una figura con due pannelli (mAP e FAR), i tre
    benchmark come curve, tutte lisciate con media mobile.

    'finestre' e' un dizionario {'map': int, 'far': int} con l'ampiezza della
    finestra per ciascuna metrica: la FAR e' tipicamente piu' rumorosa e
    beneficia di una finestra piu' larga.
    """
    if finestre is None:
        finestre = {'map': 5, 'far': 11}

    per_tipo = {re.sub(r'_s\d+$', '', r): r for r in run_da_plottare}

    tipi_presenti = [t for t in TIPI_BASE if t in per_tipo]
    if not tipi_presenti:
        print("Nessun modello base trovato tra le run rappresentative.")
        return

    for tipo in tipi_presenti:
        run_base = per_tipo[tipo]
        benchmarks = dati_plot[run_base]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            f"Training of: {run_base} ",
            fontsize=16,
        )

        for benchmark, metriche in benchmarks.items():
            dati_ordinati = sorted(zip(metriche['epochs'], metriche['map'], metriche['far']))
            epoche, maps, fars = zip(*dati_ordinati)

            maps_lisci = moving_average(list(maps), finestre['map'])
            fars_lisci = moving_average(list(fars), finestre['far'])

            ax1.plot(epoche, maps_lisci, marker='o', markersize=3, label=benchmark)
            ax2.plot(epoche, fars_lisci, marker='o', markersize=3, label=benchmark)

        ax1.set_title(f"{NOMI_METRICA['map']} vs Epochs")
        ax1.set_xlabel('Epochs')
        formatta_asse(ax1, NOMI_METRICA['map'], 'map')
        ax1.legend()

        ax2.set_title('FAR vs Epochs')
        ax2.set_xlabel('Epochs')
        formatta_asse(ax2, 'False Acceptance Rate', 'far')
        ax2.legend()

        plt.tight_layout()

        percorso = os.path.join(cartella, f'{run_base}_smooth_metrics.png')
        plt.savefig(percorso, dpi=150)
        plt.close(fig)
        print(f"Salvato: {percorso}")


def main():
    base_dir = Path(__file__).resolve().parent
    cartella_plots = base_dir / 'plots'
    cartella_plots.mkdir(parents=True, exist_ok=True)
    percorso_ledger = base_dir / 'ledger.jsonl'

    dati_plot = carica_ledger(percorso_ledger)
    run_da_plottare = trova_run_rappresentative(dati_plot)

    # finestre distinte: la FAR e' piu' rumorosa, quindi piu' larga
    plot_base_smoothed(
        dati_plot,
        run_da_plottare,
        cartella=cartella_plots,
        finestre={'map': 5, 'far': 11},
    )


if __name__ == '__main__':
    main()