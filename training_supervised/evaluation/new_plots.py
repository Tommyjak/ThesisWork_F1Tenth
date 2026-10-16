import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt

def trova_run_rappresentative(dati_plot):
    """
    Raggruppa le run per 'tipo' (ignorando il seed) e restituisce 
    per ogni tipo la run con la traiettoria globale più vicina alla media.
    """
    tipi_run = {}
    
    # 1. Raggruppa le run base in base al loro "tipo" (rimuove suffisso _s1, _s2, ecc.)
    for run_base in dati_plot.keys():
        tipo = re.sub(r'_s\d+$', '', run_base)
        if tipo not in tipi_run:
            tipi_run[tipo] = []
        tipi_run[tipo].append(run_base)
        
    run_scelte = []
    
    # 2. Calcola la distanza dalla media per ogni tipo
    for tipo, runs in tipi_run.items():
        if len(runs) == 1:
            run_scelte.append(runs[0])
            continue
            
        distanze_da_media = {r: 0.0 for r in runs}
        conteggio_valori = {r: 0 for r in runs}
        
        # Analizza i benchmark comuni a tutte le run di questo tipo
        benchmarks_comuni = set.intersection(*[set(dati_plot[r].keys()) for r in runs])
        
        for b in benchmarks_comuni:
            # Trova le epoche valutate in comune tra i seed
            epoche_comuni = set.intersection(*[set(dati_plot[r][b]['epochs']) for r in runs])
            
            for epoca in epoche_comuni:
                # Estrae i mAP di tutti i seed in questa precisa epoca
                maps_epoca = []
                for r in runs:
                    idx = dati_plot[r][b]['epochs'].index(epoca)
                    maps_epoca.append(dati_plot[r][b]['map'][idx])
                
                media_map = np.mean(maps_epoca)
                
                # Somma l'errore assoluto (distanza dalla media)
                for r in runs:
                    idx = dati_plot[r][b]['epochs'].index(epoca)
                    valore_map = dati_plot[r][b]['map'][idx]
                    distanze_da_media[r] += abs(valore_map - media_map)
                    conteggio_valori[r] += 1
                    
        # Seleziona la run con l'errore medio assoluto minimo
        run_migliore = min(runs, key=lambda r: distanze_da_media[r] / max(1, conteggio_valori[r]))
        run_scelte.append(run_migliore)
        print(f"[{tipo}] Run rappresentativa selezionata (più vicina alla media): {run_migliore}")
        
    return run_scelte

def tabella_tempi(run_da_plottare, percorso_ledger='ledger.jsonl',
                  percorso_csv=os.path.join('plots', 'tempi_training.csv')):
    """
    Stampa e salva una tabella con il tempo di training, in minuti, di ogni
    run plottata.

    Il campo 'train_s' del ledger e' il tempo cumulativo di training all'epoca
    di quel checkpoint: il valore massimo su una run e' quindi il suo costo
    totale. Si legge il ledger una seconda volta invece di portarsi dietro il
    dato: costa un decimo di secondo e tiene la funzione indipendente dalla
    struttura di dati_plot.
    """
    tempi = {}
    epoche = {}

    with open(percorso_ledger, 'r') as f:
        for linea in f:
            if not linea.strip():
                continue

            row = json.loads(linea)

            if row.get('train_s') is None:
                continue

            base = row['run'].split('@')[0]
            tempi[base] = max(tempi.get(base, 0.0), row['train_s'])
            epoche[base] = max(epoche.get(base, 0), row.get('epoch', 0))

    righe = [
        {'run': r, 'tipo': re.sub(r'_s\d+$', '', r), 'epoche': epoche[r], 'minuti': tempi[r] / 60}
        for r in sorted(run_da_plottare) if r in tempi
    ]

    if not righe:
        print("Nessun tempo di training trovato nel ledger.")
        return righe

    larghezza = max(len(r['run']) for r in righe)

    print(f"\n{'run':<{larghezza}}  {'epoche':>6}  {'minuti':>7}")
    print('-' * (larghezza + 17))

    for r in righe:
        print(f"{r['run']:<{larghezza}}  {r['epoche']:>6}  {r['minuti']:>7.1f}")

    os.makedirs(os.path.dirname(percorso_csv), exist_ok=True)

    with open(percorso_csv, 'w') as f:
        f.write('run,tipo,epoche,minuti\n')

        for r in righe:
            f.write(f"{r['run']},{r['tipo']},{r['epoche']},{r['minuti']:.1f}\n")

    print(f"\nTabella salvata in {percorso_csv}")

    return righe

# --- confronto fra i bracci, un grafico per benchmark e per metrica ----------
#
# Le figure per configurazione mostrano i tre benchmark dentro una run; queste
# mostrano le dieci run dentro un benchmark, che e' la direzione in cui sta la
# domanda della tesi: a parita' di ambiente valutato, come si comportano joint
# retrain, fine tuning e replay.
#
# Tre colori, uno per famiglia: sono un sottoinsieme di una palette verificata
# sulla separazione per daltonismo (deuteranopia e protanopia). Dentro la
# famiglia cambia lo stile della linea, non la tinta, cosi' l'appartenenza si
# legge a colpo d'occhio e il budget si legge dalla legenda.
# --- convenzioni comuni a tutte le figure ------------------------------------
#
# La metrica riportata e' mAP50-95, la media delle AP a soglie IoU da 0.50 a
# 0.95: e' lo standard COCO ed e' la stessa su cui sono calcolate le metriche
# di continual learning, cosi' figure e tabelle parlano dello stesso numero.
#
# I valori del ledger sono frazioni; qui vengono moltiplicati per 100 una volta
# sola, in lettura, e da li' in poi tutto e' in punti percentuali.
COLONNA_MAP = 'all/mAP50_95'
COLONNA_FAR = 'negatives/far'
NOMI_METRICA = {'map': 'mAP50-95', 'far': 'FAR'}
SCALA = 100.0

# Asse y fisso 0-100 su ogni figura: due grafici qualsiasi si confrontano senza
# leggere la scala. Le etichette sono ripetute a destra perche' su una figura
# larga il valore di fine curva e' lontano dall'asse sinistro.
# Il mAP occupa tutta la scala, il FAR vive sotto il 20%: una scala unica
# renderebbe il secondo illeggibile. Restano due scale fisse e dichiarate, non
# due autoscale, quindi due figure della stessa metrica sono confrontabili.
LIMITE_Y = {'map': (0, 100), 'far': (0, 20)}

def formatta_asse(ax, nome_metrica, chiave):
    ax.set_ylim(*LIMITE_Y[chiave])
    ax.set_ylabel(f'{nome_metrica} (%)')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='y', which='both', left=True, right=True,
                   labelleft=True, labelright=True)


def segnala_fuori_scala(ax, massimo, chiave):
    """
    Con una scala fissa una curva che esce dal riquadro viene tagliata dal
    bordo, e un lettore la legge come dato mancante invece che come valore
    alto: l'errore e' peggiore del problema. Qui la figura dichiara il taglio.
    """
    limite = LIMITE_Y[chiave][1]

    if massimo is None or massimo <= limite:
        return


COLORI_FAMIGLIA = {
    'joint': "#0004ed",
    'ft':    "#ff0000",
    'ftr':   "#009E00",
}

STILI = ['-', '--', '-.', ':']

# ordine di presentazione: crescente per quantita' di dati dentro ogni famiglia
ORDINE_CONFRONTO = [
    'joint_w2', 'joint_w10', 'joint_w50', 'joint_w100',
    'ft_d10', 'ft_d50', 'ft',
    'ftr_02', 'ftr_10', 'ftr_50',
]

ETICHETTE = {
    'joint_w2': 'joint 2%',   'joint_w10': 'joint 10%',
    'joint_w50': 'joint 50%', 'joint_w100': 'joint 100%',
    'ft_d10': 'ft 10% dati',  'ft_d50': 'ft 50% dati', 'ft': 'ft 100% dati',
    'ftr_02': 'replay 2%',    'ftr_10': 'replay 10%',  'ftr_50': 'replay 50%',
}

RUOLI = {
    'icra_holdout':        'specialization',
    'FIM_outside_holdout': 'forgetting',
    'FIM_inside':          'generalization',
}


def famiglia(tipo):
    if tipo.startswith('joint'):
        return 'joint'

    if tipo.startswith('ftr'):
        return 'ftr'

    return 'ft'


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


# --- la terna fissa ----------------------------------------------------------
#
# I tre modelli del confronto NON sono scelti sui risultati: sono determinati
# dal disegno sperimentale. Tutti e tre vedono il 100% del nuovo dominio, e
# differiscono solo per cosa fanno dello storico. Letti in quest'ordine ogni
# passaggio cambia una variabile sola:
#
#   ft         100% nuovo,  0% storico, parte dal base   -- niente replay
#   ftr_10     100% nuovo, 10% storico, parte dal base   -- aggiunge il buffer
#   joint_w10  100% nuovo, 10% storico, parte da zero    -- stessi identici dati
#
# ft -> ftr_10 isola l'effetto del replay; ftr_10 -> joint_w10 isola il costo di
# riaddestrare da zero, perche' quei due usano le stesse immagini.
#
# Il 10% e' il livello in cui la curva ritenzione/costo del replay gira, ed e'
# quindi la configurazione che il lavoro di fatto raccomanda.
TERNA = ['ft', 'ftr_10', 'joint_w10']

# joint al 100% dello storico: non e' un pari degli altri tre, e' il soffitto
# contro cui si leggono. Disegnato sottile e grigio per dire che e' riferimento.
UPPER_BOUND = 'joint_w100'

# il modello di partenza di tutti i bracci: la sua prestazione finale e' la
# linea rispetto a cui l'oblio e' un'area, non un numero astratto
TIPO_BASE = 'base_50_50'

STILE_TERNA = {
    'ft':        {'color': COLORI_FAMIGLIA['ft'],
                  'label': 'naive fine tuning'},
    'ftr_10':    {'color': COLORI_FAMIGLIA['ftr'],
                  'label': 'replay, 10% buffer'},
    'joint_w10': {'color': COLORI_FAMIGLIA['joint'],
                  'label': 'joint retrain, 10%'},
}


def valore_finale(dati_plot, run, benchmark, chiave):
    """Il valore all'ultima epoca valutata, o None."""
    c = curva(dati_plot, run, benchmark, chiave)

    return None if c is None else c[1][-1]


def plot_confronto(dati_plot, run_da_plottare, cartella='plots'):
    """
    Una figura per ogni combinazione (benchmark, metrica): la terna fissa, piu'
    il joint al 100% come upper bound e la linea orizzontale del modello base.

    Le curve corte (30 epoche) sono traslate a destra in modo che il loro
    ultimo punto cada sull'ultimo punto delle curve lunghe (100 epoche): si
    confrontano cosi' gli stati finali invece di lasciare due terzi della
    figura ai soli joint. L'asse x smette di essere un'epoca assoluta, e per
    questo l'etichetta lo dichiara e la legenda riporta lo shift.
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

    for benchmark in benchmarks:
        for chiave, nome_metrica in NOMI_METRICA.items():
            curve = [(t, curva(dati_plot, per_tipo[t], benchmark, chiave)) for t in tipi]
            curve = [(t, c) for t, c in curve if c is not None]

            if not curve:
                continue

            # upper bound e linea base: riferimenti, non membri della terna
            ub = (curva(dati_plot, per_tipo[UPPER_BOUND], benchmark, chiave)
                  if UPPER_BOUND in per_tipo else None)
            base = (valore_finale(dati_plot, per_tipo[TIPO_BASE], benchmark, chiave)
                    if TIPO_BASE in per_tipo else None)

            fine = max([c[0][-1] for _, c in curve] + ([ub[0][-1]] if ub else []))

            fig, ax = plt.subplots(figsize=(10, 6))
            massimo = None

            if ub is not None:
                shift = fine - ub[0][-1]
                ax.plot([e + shift for e in ub[0]], ub[1],
                        color='#999999', linestyle='--', linewidth=1.2,
                        label='joint 100% — upper bound')
                massimo = max(massimo or 0, max(ub[1]))

            for t, (epoche, valori) in curve:
                shift = fine - epoche[-1]
                stile = STILE_TERNA[t]
                etichetta = stile['label']

                ax.plot([e + shift for e in epoche], valori,
                        color=stile['color'], linestyle='-', linewidth=2.0,
                        label=etichetta)

                massimo = max(massimo or 0, max(valori))

                print(f"[{benchmark}/{chiave}] {t}: finale {valori[-1]:.2f}, shift +{shift}")

            ax.set_xlabel(f'Epochs')
            ax.set_title(f'{nome_metrica} vs Epochs — {benchmark}\n{RUOLI[benchmark]}', fontsize=13)
            formatta_asse(ax, nome_metrica, chiave)
            segnala_fuori_scala(ax, massimo, chiave)
            ax.legend(fontsize=9, frameon=False, loc='best')

            plt.tight_layout()

            percorso = os.path.join(cartella, f'confronto_{benchmark}_{chiave}.png')
            plt.savefig(percorso, dpi=150)
            plt.close(fig)

            print(f"Salvato confronto: {percorso}")


# --- ESECUZIONE PRINCIPALE ---

os.makedirs('plots', exist_ok=True)
dati_plot = {}

with open('ledger.jsonl', 'r') as f:
    for linea in f:
        if not linea.strip():
            continue
        
        row = json.loads(linea)
        
        epoca = row.get('epoch')
        if epoca is None:
            continue

        run_grezza = row.get('run', 'sconosciuta')
        run_base = run_grezza.split('@')[0]
        
        benchmark = row.get('benchmark', 'sconosciuto')
        mAP = row.get(COLONNA_MAP)
        far = row.get(COLONNA_FAR)

        if mAP is None or far is None:
            continue

        mAP = mAP * SCALA
        far = far * SCALA

        if run_base not in dati_plot:
            dati_plot[run_base] = {}
        if benchmark not in dati_plot[run_base]:
            dati_plot[run_base][benchmark] = {'epochs': [], 'map': [], 'far': []}

        dati_plot[run_base][benchmark]['epochs'].append(epoca)
        dati_plot[run_base][benchmark]['map'].append(mAP)
        dati_plot[run_base][benchmark]['far'].append(far)

# Filtra mantenendo solo le run più rappresentative per ogni architettura/setup
run_da_plottare = trova_run_rappresentative(dati_plot)

# quanto e' costata ogni run che finisce nei grafici
tabella_tempi(run_da_plottare)

# confronto fra bracci: 3 benchmark x 2 metriche = 6 figure da 10 curve
plot_confronto(dati_plot, run_da_plottare)

# Generazione dei grafici filtrati
for run_base in run_da_plottare:
    benchmarks = dati_plot[run_base]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Training of: {run_base}', fontsize=16)

    max_map = None
    max_far = None

    for benchmark, metriche in benchmarks.items():
        # Ordina cronologicamente in base all'epoca
        dati_ordinati = sorted(zip(metriche['epochs'], metriche['map'], metriche['far']))
        epoche, maps, fars = zip(*dati_ordinati)

        ax1.plot(epoche, maps, marker='o', label=benchmark)
        ax2.plot(epoche, fars, marker='o', label=benchmark)

        max_map = max(maps) if max_map is None else max(max_map, max(maps))
        max_far = max(fars) if max_far is None else max(max_far, max(fars))

    ax1.set_title(f"{NOMI_METRICA['map']} vs Epochs")
    ax1.set_xlabel('Epochs')
    formatta_asse(ax1, NOMI_METRICA['map'], 'map')
    segnala_fuori_scala(ax1, max_map, 'map')
    ax1.legend()

    ax2.set_title('FAR vs Epochs')
    ax2.set_xlabel('Epochs')
    formatta_asse(ax2, 'False Acceptance Rate', 'far')
    segnala_fuori_scala(ax2, max_far, 'far')
    ax2.legend()

    plt.tight_layout()
    
    percorso_salvataggio = os.path.join('plots', f'{run_base}_metrics.png')
    plt.savefig(percorso_salvataggio)
    plt.close()
    
    print(f"Salvato plot per la run: {percorso_salvataggio}")