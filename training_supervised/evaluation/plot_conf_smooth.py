"""
confronto_smooth.py
-------------------
Genera i grafici di confronto fra bracci (joint / fine tuning / replay),
uno per ogni combinazione (benchmark, metrica), con le curve lisciate tramite
media mobile. Versione autonoma: legge il ledger, seleziona la run piu'
rappresentativa per tipo e disegna la terna fissa piu' l'upper bound.

Uso:
    python confronto_smooth.py
"""

import json
import os
import re

import numpy as np
import matplotlib.pyplot as plt


# --- convenzioni e configurazione --------------------------------------------
COLONNA_MAP = 'all/mAP50_95'
COLONNA_FAR = 'negatives/far'
NOMI_METRICA = {'map': 'mAP50-95', 'far': 'FAR'}
SCALA = 100.0

# scale y fisse e dichiarate: due figure della stessa metrica sono confrontabili
LIMITE_Y = {'map': (0, 100), 'far': (0, 20)}

COLORI_FAMIGLIA = {
    'joint': "#0004ed",
    'ft':    "#ff0000",
    'ftr':   "#009E00",
}

RUOLI = {
    'icra_holdout':        'specialization',
    'FIM_outside_holdout': 'forgetting',
    'FIM_inside':          'generalization',
}

# la terna fissa: tutti vedono il 100% del nuovo dominio, differiscono solo
# per cosa fanno dello storico. ft -> ftr_10 isola il replay; ftr_10 ->
# joint_w10 isola il costo di riaddestrare da zero (stessi identici dati).
TERNA = ['ft', 'ftr_10', 'joint_w10']
UPPER_BOUND = 'joint_w100'   # soffitto di riferimento, non membro della terna
TIPO_BASE = 'base_50_50'     # modello di partenza dei bracci

STILE_TERNA = {
    'ft':        {'color': COLORI_FAMIGLIA['ft'],  'label': 'naive fine tuning'},
    'ftr_10':    {'color': COLORI_FAMIGLIA['ftr'], 'label': 'replay, 10% buffer'},
    'joint_w10': {'color': COLORI_FAMIGLIA['joint'], 'label': 'joint retrain, 10%'},
}


# --- utility -----------------------------------------------------------------
def formatta_asse(ax, nome_metrica, chiave):
    ax.set_ylim(*LIMITE_Y[chiave])
    ax.set_ylabel(f'{nome_metrica} (%)')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='y', which='both', left=True, right=True,
                   labelleft=True, labelright=True)


def moving_average(valori, finestra=5):
    """
    Media mobile centrata su 'finestra' epoche. Restituisce una lista lunga
    quanto l'input: ai bordi la finestra si riduce ai punti disponibili, cosi'
    la curva liscia parte e finisce agli stessi estremi di quella grezza
    (necessario qui, perche' il confronto avviene sugli stati finali).
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


def curva(dati_plot, run, benchmark, chiave):
    """Serie (epoche, valori) ordinata per epoca, o None se non e' plottabile."""
    d = dati_plot.get(run, {}).get(benchmark)
    if not d:
        return None

    punti = sorted(zip(d['epochs'], d[chiave]))
    if len(punti) < 2:
        return None

    epoche, valori = zip(*punti)
    return list(epoche), list(valori)


def valore_finale(dati_plot, run, benchmark, chiave):
    """Il valore all'ultima epoca valutata, o None."""
    c = curva(dati_plot, run, benchmark, chiave)
    return None if c is None else c[1][-1]


# --- selezione delle run rappresentative -------------------------------------
def trova_run_rappresentative(dati_plot):
    """
    Raggruppa le run per 'tipo' (ignorando il seed _sN) e restituisce, per ogni
    tipo, la run con la traiettoria globale piu' vicina alla media dei seed.
    """
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
                maps_epoca = []
                for r in runs:
                    idx = dati_plot[r][b]['epochs'].index(epoca)
                    maps_epoca.append(dati_plot[r][b]['map'][idx])
                media = np.mean(maps_epoca)
                for r in runs:
                    idx = dati_plot[r][b]['epochs'].index(epoca)
                    distanze[r] += abs(dati_plot[r][b]['map'][idx] - media)
                    conteggi[r] += 1

        migliore = min(runs, key=lambda r: distanze[r] / max(1, conteggi[r]))
        run_scelte.append(migliore)
        print(f"[{tipo}] Run rappresentativa: {migliore}")

    return run_scelte


# --- plotting con smoothing --------------------------------------------------
def plot_confronto_smoothed(dati_plot, run_da_plottare, cartella='plots', finestra=5):
    """
    Una figura per ogni combinazione (benchmark, metrica): la terna fissa piu'
    il joint 100% come upper bound, con le curve lisciate da media mobile.

    Le curve corte (30 epoche) sono traslate a destra in modo che il loro
    ultimo punto cada sull'ultimo punto delle curve lunghe (100 epoche): si
    confrontano gli stati finali. Lo smoothing e' applicato ai soli valori,
    dopo l'allineamento, quindi lo stato finale resta ancorato al dato reale.
    """
    per_tipo = {re.sub(r'_s\d+$', '', r): r for r in run_da_plottare}

    mancanti = [t for t in TERNA if t not in per_tipo]
    if mancanti:
        print(f"Attenzione: mancano dal ledger {mancanti}, la terna sara' incompleta.")

    tipi = [t for t in TERNA if t in per_tipo]
    if not tipi:
        print("Nessun braccio da confrontare.")
        return

    benchmarks = [b for b in RUOLI if any(b in dati_plot[per_tipo[t]] for t in tipi)]

    FINESTRA = {'map': 5, 'far': 11}

    for benchmark in benchmarks:
        for chiave, nome_metrica in NOMI_METRICA.items():
            finestra = FINESTRA[chiave]
            curve = [(t, curva(dati_plot, per_tipo[t], benchmark, chiave)) for t in tipi]
            curve = [(t, c) for t, c in curve if c is not None]
            if not curve:
                continue

            ub = (curva(dati_plot, per_tipo[UPPER_BOUND], benchmark, chiave)
                  if UPPER_BOUND in per_tipo else None)

            fine = max([c[0][-1] for _, c in curve] + ([ub[0][-1]] if ub else []))

            fig, ax = plt.subplots(figsize=(10, 6))

            if ub is not None:
                shift = fine - ub[0][-1]
                lisci = moving_average(ub[1], finestra)
                ax.plot([e + shift for e in ub[0]], lisci,
                        color='#999999', linestyle='--', linewidth=1.2,
                        label='joint 100% — upper bound')

            for t, (epoche, valori) in curve:
                shift = fine - epoche[-1]
                stile = STILE_TERNA[t]
                lisci = moving_average(valori, finestra)
                ax.plot([e + shift for e in epoche], lisci,
                        color=stile['color'], linestyle='-', linewidth=2.0,
                        label=stile['label'])
                print(f"[{benchmark}/{chiave}] {t}: finale {lisci[-1]:.2f}, shift +{shift}")

            ax.set_xlabel('Epochs')
            ax.set_title(f'{nome_metrica} vs Epochs — {benchmark}\n'
                         f'{RUOLI[benchmark]}', fontsize=13)
            formatta_asse(ax, nome_metrica, chiave)
            ax.legend(fontsize=9, frameon=False, loc='best')

            plt.tight_layout()
            percorso = os.path.join(cartella, f'confronto_smooth_{benchmark}_{chiave}.png')
            plt.savefig(percorso, dpi=150)
            plt.close(fig)
            print(f"Salvato: {percorso}")


# --- caricamento del ledger --------------------------------------------------
def carica_ledger(percorso='ledger.jsonl'):
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


# --- esecuzione principale ---------------------------------------------------
def main():
    os.makedirs('plots', exist_ok=True)
    dati_plot = carica_ledger('ledger.jsonl')
    run_da_plottare = trova_run_rappresentative(dati_plot)
    plot_confronto_smoothed(dati_plot, run_da_plottare, finestra=5)


if __name__ == '__main__':
    main()