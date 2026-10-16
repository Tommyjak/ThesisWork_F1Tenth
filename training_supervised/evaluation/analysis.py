import pandas as pd
from pathlib import Path

LEDGER_PATH = Path(__file__).resolve().parent / "ledger.jsonl"

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', None)

COLUMNS = [
    'all/mAP50',
    'all/mAP50_95',
    'positives/recall',
    'negatives/far',
    'negatives/fp_per_neg_image',
]

def load_ledger(ledger_path = LEDGER_PATH):
    if not Path(ledger_path).is_file():
        raise FileNotFoundError(f"Ledger non trovato: {ledger_path}. Lancia prima benchmark.py")

    return pd.read_json(ledger_path, lines = True)

def per_run(df, benchmark, columns = COLUMNS):
    selected = df[df['benchmark'] == benchmark]
    present = [c for c in columns if c in selected.columns]

    return selected.set_index('run')[present].round(4)

def per_arm(df, benchmark, column):
    selected = df[df['benchmark'] == benchmark]

    return selected.pivot_table(index = 'arm', columns = 'tag', values = column).round(4)

# quanto ogni braccio perde rispetto al modello che non puo' aver dimenticato
# nulla: sul benchmark di forgetting e' la misura dell'oblio
def forgetting(df, benchmark, column = 'all/mAP50', reference = 'base'):
    table = per_arm(df, benchmark, column)

    if reference not in table.index:
        raise KeyError(f"Braccio di riferimento '{reference}' assente: {list(table.index)}")

    return (table.loc[reference] - table).round(4)


# --- metriche di continual learning -------------------------------------------
#
# Nomenclatura standard (Lopez-Paz & Ranzato 2017 per BWT/FWT, Chaudhry et al.
# 2018 per forgetting e intransigence), definita sulla matrice R[i][j]: la
# performance sul dominio j dopo aver allenato fino al passo i.
#
# Qui i domini sono due (vecchio -> icra), quindi la matrice ha una riga sola di
# "dopo" e ogni scalare collassa in una differenza fra due righe del ledger. Non
# si costruisce nessuna matrice esplicita: sarebbe sovrastruttura per T=2.
#
#   R[1][1] = base_<tag>  su FIM_outside_holdout      R[1][2] = base_<tag>  su icra_holdout
#   R[2][1] = <arm>_<tag> su FIM_outside_holdout      R[2][2] = <arm>_<tag> su icra_holdout
#   R*[j]   = retrain_<tag> (joint), l'upper bound
#   b[j]    = yolo26n, il modello mai allenato

# il dominio vecchio non e' omogeneo: f1tenth vale il 54% dei suoi positivi,
# FIM_outside il 25%, squeaky il 20%. FIM_outside_holdout e' una sonda sola, e
# per giunta la meno severa (stessa giornata, stesso rig, stessa location del
# suo train): misura l'oblio a spostamento di dominio quasi nullo.
# squeaky_holdout e' l'altra giornata, l'altro setup, e ha 4 gruppi -> serve a
# distinguere "ha dimenticato la sessione" da "ha dimenticato il dominio"
BM_OLD = 'FIM_outside_holdout'      # sonda primaria, retro-compatibile
BM_OLD_ALL = ['FIM_outside_holdout', 'squeaky_holdout']
BM_NEW = 'icra_holdout'             # dominio nuovo: la sonda dell'adattamento
BM_UNSEEN = 'FIM_inside'            # mai allenato: generalizzazione, fuori matrice

ARM_BASELINE = 'baseline'           # il modello mai allenato, il b[j] del FWT

# con piu' taglie di modello a benchmark ci sono PIU' righe con arm 'baseline'
# (yolo26n, yolo26s, ...) e selezionarle per solo arm farebbe alzare a scalar()
# "Cella ambigua", fermando tutta l'analisi. Il forward transfer va misurato
# contro il modello non allenato della STESSA taglia dei bracci di quel tag
BASELINE_RUN = 'yolo26n'
BASELINE_BY_TAG = {'s': 'yolo26s', 'm': 'yolo26m', 's@last': 'yolo26s', 'm@last': 'yolo26m'}

# il modello del passo 1 da cui partono i bracci di un tag. Per i tag 's' ed 'm'
# la base porta gia' quel tag e la selezione per (arm, tag) basta; per 'main'
# no: i bracci principali partono da base_balanced, che porta tag 'balanced'
# perche' e' anche un punto dell'asse dei negativi. Senza questa mappa la riga
# 'base' della tabella principale sarebbe vuota e con essa TUTTA la colonna
# forgetting, che e' una differenza rispetto a quel valore
# campagna definitiva: la base e' base_50_50_s2 - configurazione "full" (tutti
# i 4603 negativi), replica mediana su mAP50 fra le tre. Il pilota usava
# base_balanced, che e' risultata un seed fortunato
# la tabella @last usa LO STESSO riferimento della tabella best: i bracci sono
# partiti da base_50_50_s2/best.pt (epoca 90), e l'oblio si misura rispetto al
# checkpoint da cui si e' davvero partiti, non rispetto a last.pt della base
BASE_BY_TAG = {'main': 'base_50_50_s2', 'main@last': 'base_50_50_s2'}

# le repliche della base: stessi dati, stessa composizione, stesso split, solo
# il seed di training cambia. base_balanced ne ha quattro perche' il suo valore
# su icra_holdout (0.2433) e' risultato a 4.4 deviazioni dalla media delle altre
# tre - un seed fortunato, non un modello migliore
BASE_REPLICAS = {
    'main': ['base_50_50_s1', 'base_50_50_s2', 'base_50_50_s3'],
}

# le quattro configurazioni della prima tranche, tre repliche ciascuna. Servono
# alla sezione sulle basi: e' l'esperimento sull'asse dei negativi
BASE_CONFIGS = {
    '90_10': ['base_90_10_s1', 'base_90_10_s2', 'base_90_10_s3'],
    '70_30': ['base_70_30_s1', 'base_70_30_s2', 'base_70_30_s3'],
    'full':  ['base_50_50_s1', 'base_50_50_s2', 'base_50_50_s3'],
    'hard':  ['base_hard_s1',  'base_hard_s2',  'base_hard_s3'],
}
ARM_BASE = 'base'                   # il modello dopo il passo 1
ARM_JOINT = 'joint'                 # retrain, l'upper bound R*
ARM_STEP2 = ('joint', 'fine_tune', 'replay')

MAP = 'all/mAP50'
MAP95 = 'all/mAP50_95'
FAR = 'negatives/far'

CL_COLUMNS = [
    'mAP/old',
    'mAP95/old',
    'mAP/new',
    'far/old',
    'forgetting/mAP50',
    'forgetting/mAP50_95',
    'forgetting/far',
    'intransigence',
    'ACC',
]

# una sola cella della matrice. Se la combinazione non e' nel ledger torna None
# invece di sollevare: un benchmark non ancora girato deve lasciare un buco nella
# tabella, non far fallire tutta l'analisi
def scalar(df, benchmark, column, arm = None, tag = None, run = None):
    if column not in df.columns:
        return None

    selected = df[df['benchmark'] == benchmark]

    if run is not None:
        selected = selected[selected['run'] == run]

    if arm is not None:
        selected = selected[selected['arm'] == arm]

    if tag is not None:
        selected = selected[selected['tag'] == tag]

    values = selected[column].dropna()

    if values.empty:
        return None

    # due righe per la stessa cella significano un ledger sporco (tipicamente
    # run di una versione precedente rimaste li'): meglio fermarsi che mediare
    if len(values) > 1:
        raise ValueError(
            f"Cella ambigua: {len(values)} righe per benchmark={benchmark} "
            f"arm={arm} tag={tag} run={run} -> {list(selected['run'])}"
        )

    return float(values.iloc[0])

# sulla mAP piu' alto e' meglio, sulla FAR piu' alto e' peggio. Le due
# differenze sono orientate in modo opposto cosi' che un valore positivo
# significhi sempre "peggiorato rispetto al riferimento"
def degradation(reference, current, higher_is_better = True):
    if reference is None or current is None:
        return None

    if higher_is_better:
        return reference - current

    return current - reference

# ACC e' la media su TUTTI i domini: se ne manca uno il numero non e' ACC, e
# spacciare la media del sottoinsieme presente sarebbe peggio di un buco
def mean_all(values):
    for value in values:
        if value is None:
            return None

    return sum(values) / len(values)

# i tag per cui esiste almeno un braccio di secondo passo: sono gli unici su cui
# le metriche di continual learning hanno senso
EPOCH_SUFFIX = '@ep'

# le righe dei checkpoint intermedi (tag "<tag>@ep", una per epoca salvata):
# non sono celle della matrice, sono punti di una traiettoria. Vanno lette a
# parte; nelle tabelle per (arm, tag) farebbero alzare "Cella ambigua"
def is_epoch_row(df):
    if 'checkpoint' in df.columns:
        return df['checkpoint'] == 'epoch'

    return df['tag'].fillna('').str.endswith(EPOCH_SUFFIX) if 'tag' in df.columns else pd.Series(False, index = df.index)

def final_only(df):
    return df[~is_epoch_row(df)]

def cl_tags(df):
    if 'arm' not in df.columns or 'tag' not in df.columns:
        return []

    selected = final_only(df)[final_only(df)['arm'].isin(ARM_STEP2)]

    return sorted(selected['tag'].dropna().unique())

def cl_metrics(df, tag, bm_old = BM_OLD):
    base_run = BASE_BY_TAG.get(tag)
    base_tag = None if base_run else tag

    base_map_old = scalar(df, bm_old, MAP, arm = ARM_BASE, tag = base_tag, run = base_run)
    base_map95_old = scalar(df, bm_old, MAP95, arm = ARM_BASE, tag = base_tag, run = base_run)
    base_far_old = scalar(df, bm_old, FAR, arm = ARM_BASE, tag = base_tag, run = base_run)
    joint_map_new = scalar(df, BM_NEW, MAP, arm = ARM_JOINT, tag = tag)

    rows = {}

    for arm in (ARM_BASE,) + ARM_STEP2:
        # la riga 'base' puo' vivere sotto un tag diverso da quello dei bracci
        run = base_run if arm == ARM_BASE else None
        row_tag = base_tag if arm == ARM_BASE else tag

        map_old = scalar(df, bm_old, MAP, arm = arm, tag = row_tag, run = run)
        map95_old = scalar(df, bm_old, MAP95, arm = arm, tag = row_tag, run = run)
        map_new = scalar(df, BM_NEW, MAP, arm = arm, tag = row_tag, run = run)
        far_old = scalar(df, bm_old, FAR, arm = arm, tag = row_tag, run = run)

        if map_old is None and map_new is None and far_old is None:
            continue

        rows[arm] = {
            'mAP/old': map_old,
            'mAP95/old': map95_old,
            'mAP/new': map_new,
            'far/old': far_old,
            # quanto si perde sul dominio vecchio rispetto a chi non lo ha mai
            # abbandonato. E' -BWT: positivo = ha dimenticato
            'forgetting/mAP50': degradation(base_map_old, map_old, True),
            'forgetting/far': degradation(base_far_old, far_old, False),
            # la mAP50 nasconde il danno alla localizzazione: nella prima
            # tornata il fine-tuning perdeva 5.6 punti di mAP50 e 16.7 di
            # mAP50-95 sulla stessa sonda. Le due colonne vanno lette insieme
            'forgetting/mAP50_95': degradation(base_map95_old, map95_old, True),
            # quanto NON si riesce a imparare il dominio nuovo rispetto al joint.
            # Senza questa colonna un modello congelato sembrerebbe ottimo: zero
            # forgetting solo perche' non ha imparato niente
            'intransigence': degradation(joint_map_new, map_new, True),
            'ACC': mean_all([map_old, map_new]),
        }

    table = pd.DataFrame.from_dict(rows, orient = 'index')
    table.index.name = 'arm'

    return table.reindex(columns = CL_COLUMNS).round(4)

# forward transfer: quanto il dominio vecchio aiuta gia' sul nuovo, a costo zero,
# misurato contro il modello mai allenato. Fuori dalla tabella perche' riguarda
# il passo 1, non i bracci di continual learning
def transfer(df, tag):
    baseline_run = BASELINE_BY_TAG.get(tag, BASELINE_RUN)
    base_run = BASE_BY_TAG.get(tag)

    baseline_new = scalar(df, BM_NEW, MAP, arm = ARM_BASELINE, run = baseline_run)
    base_new = scalar(df, BM_NEW, MAP, arm = ARM_BASE,
                      tag = None if base_run else tag, run = base_run)

    rows = {
        'FWT': degradation(baseline_new, base_new, False),
        f"{BM_NEW}: {baseline_run}": baseline_new,
        f"{BM_NEW}: {base_run or 'base_' + str(tag)}": base_new,
    }

    # con le repliche della base il FWT su una run sola non e' riportabile: su
    # base_balanced vale 0.2403, sulla media delle quattro repliche 0.1508 con
    # una deviazione di 0.0637. La differenza non e' un dettaglio, e' l'intero
    # effetto. Il valore singolo resta sopra perche' descrive il checkpoint da
    # cui i bracci sono davvero partiti
    replicas = BASE_REPLICAS.get(tag)

    if replicas:
        mean, sd, n = mean_sd(df, BM_NEW, MAP, replicas)

        if mean is not None:
            rows['FWT (media repliche)'] = degradation(baseline_new, mean, False)
            rows['  media base'] = mean
            rows['  sd base'] = sd
            rows['  n repliche'] = n

    return pd.Series(rows, name = tag).round(4)

# FIM_inside non e' allenato da nessun braccio: non misura oblio ma tenuta su un
# ambiente mai visto. Va letto a parte, non dentro la matrice
def generalization(df, columns = COLUMNS):
    return per_run(df, BM_UNSEEN, columns)


# --- le curve ------------------------------------------------------------------
#
# I ladder della nuova griglia (memoria del replay, dati nuovi, peso della
# miscela joint, negativi di icra_base) usano arm dedicati - *_sweep, seed_rep -
# che cl_metrics ignora di proposito: una curva non e' una cella di una matrice
# T x T, e infilarla nella tabella produrrebbe righe per meta' vuote.
#
# Le curve si indicizzano per NOME DI RUN, non per tag: il punto a x=0.10 della
# curva di memoria e' ftr_10, che pero' porta tag "main" perche' e' anche la
# cella canonica del replay. Legarle al nome e' l'unico modo di farle coesistere.

CURVES = {
    # i punti della curva sono il seed 1; media e sd su tutti e tre stanno in
    # seeds(). Le frazioni coincidono fra joint e replay di proposito: a parita'
    # di x le due curve vedono gli stessi identici dati
    'memoria del replay  (quota del dominio vecchio conservata)': [
        ('ftr_02_s1', 0.02),
        ('ftr_10_s1', 0.10),
        ('ftr_50_s1', 0.50),
    ],
    'peso della miscela joint  (quota del dominio vecchio, da zero)': [
        ('joint_w2_s1',   0.02),
        ('joint_w10_s1',  0.10),
        ('joint_w50_s1',  0.50),
        ('joint_w100_s1', 1.00),
    ],
    'dati nuovi  (frazione di icra_base)': [
        ('ft_d10_s1', 0.10),
        ('ft_d50_s1', 0.50),
        ('ft_s1',     1.00),
    ],
}

# le coppie appaiate: stessi dati, punto di partenza diverso. E' il confronto
# "riparto da zero o continuo dal checkpoint?" a parita' esatta di immagini
PAIRED = [
    (0.02, 'joint_w2_s1',   'ftr_02_s1'),
    (0.10, 'joint_w10_s1',  'ftr_10_s1'),
    (0.50, 'joint_w50_s1',  'ftr_50_s1'),
]

# le repliche di seed: stesso braccio, stessi dati, solo il seed cambia.
# Vanno aggregate, mai messe in tabella come run distinte
SEED_GROUPS = {
    stem: [f"{stem}_s1", f"{stem}_s2", f"{stem}_s3"]
    for stem in (
        'base_90_10', 'base_70_30', 'base_50_50', 'base_hard',
        'joint_w100', 'joint_w50', 'joint_w10', 'joint_w2',
        'ft', 'ft_d50', 'ft_d10',
        'ftr_50', 'ftr_10', 'ftr_02',
    )
}

CURVE_COLUMNS = [MAP, MAP95, FAR]

# --- il disegno annidato della FASE 12 ----------------------------------------
#
# Le repliche di ft e ftr_10 esistenti variano SOLO il seed del fine-tuning:
# partono tutte dallo stesso base_balanced. Le loro deviazioni non contengono
# quindi la varianza della base, che su icra_holdout e' la componente dominante
# (sd 0.0637 fra le quattro repliche della base contro 0.0090 fra i seed di ft).
# Le run from_s* variano la base tenendo fisso il seed: le due righe insieme
# separano le due sorgenti invece di confonderle in una barra sola.

NESTED_GROUPS = {
    # nella campagna definitiva i bracci partono tutti da base_50_50_s2: il
    # disegno annidato (varia la base a seed fisso) non e' stato incluso, quindi
    # questa sezione resta vuota. La definizione e' qui per quando servira'
}

# media, deviazione e numerosita' su un insieme di run. La deviazione su una
# replica sola e' None, non zero: non e' stimabile
def mean_sd(df, benchmark, column, runs):
    values = []

    for run in runs:
        value = scalar(df, benchmark, column, run = run)

        if value is not None:
            values.append(value)

    if not values:
        return None, None, 0

    series = pd.Series(values)

    return float(series.mean()), (float(series.std()) if len(values) > 1 else None), len(values)

# quanto dell'incertezza viene dalla base e quanto dal fine-tuning. Se la riga
# "varia la base" e' molto piu' larga dell'altra, le barre d'errore riportate
# finora sottostimavano l'incertezza vera
def nested(df, benchmark, column = MAP95):
    rows = {}

    for arm, sources in NESTED_GROUPS.items():
        for label, runs in sources.items():
            mean, sd, n = mean_sd(df, benchmark, column, runs)

            if n == 0:
                continue

            rows[(arm, label)] = {'media': mean, 'sd': sd, 'n': n}

    if not rows:
        return None

    table = pd.DataFrame.from_dict(rows, orient = 'index')
    table.index = pd.MultiIndex.from_tuples(table.index, names = ['arm', 'sorgente'])

    return table.round(4)

# una curva su un benchmark: x sulle righe, metriche sulle colonne. I punti non
# ancora allenati semplicemente non compaiono, cosi' la curva si puo' guardare
# a meta' esperimento
def curve(df, points, benchmark, columns = CURVE_COLUMNS):
    rows = {}

    for run, x in points:
        values = {}
        found = False

        for column in columns:
            value = scalar(df, benchmark, column, run = run)
            values[column] = value

            if value is not None:
                found = True

        if found:
            rows[x] = {'run': run, **values}

    if not rows:
        return None

    table = pd.DataFrame.from_dict(rows, orient = 'index').sort_index()
    table.index.name = 'x'

    return table.round(4)

# il ginocchio: il primo x oltre il quale il guadagno per raddoppio di x scende
# sotto una soglia. E' il numero che la tesi deve dichiarare ("oltre il X% il
# guadagno si annulla"), non una lettura a occhio del grafico
def knee(table, column = MAP, min_gain = 0.005):
    if table is None or column not in table.columns:
        return None

    points = table[column].dropna()

    if len(points) < 2:
        return None

    xs = list(points.index)

    for i in range(1, len(xs)):
        gain = points.iloc[i] - points.iloc[i - 1]

        if gain < min_gain:
            return xs[i - 1]

    return xs[-1]

# media e deviazione sulle repliche di seed. Con una sola replica la deviazione
# e' NaN, ed e' giusto: non e' zero, e' non stimabile
def seeds(df, benchmark, columns = CURVE_COLUMNS):
    rows = {}

    for label, runs in SEED_GROUPS.items():
        for column in columns:
            values = []

            for run in runs:
                value = scalar(df, benchmark, column, run = run)

                if value is not None:
                    values.append(value)

            if not values:
                continue

            series = pd.Series(values)
            rows.setdefault(label, {})[f"{column} mean"] = series.mean()
            rows.setdefault(label, {})[f"{column} std"] = series.std()
            rows.setdefault(label, {})['n'] = len(values)

    if not rows:
        return None

    return pd.DataFrame.from_dict(rows, orient = 'index').round(4)

# il costo, che e' la colonna senza la quale "meglio retrain o fine-tuning?" non
# ha risposta: il joint vince quasi sempre, se la memoria e' gratis. Non sta nel
# ledger perche' non e' una metrica di predizione: si legge dalle reference
def cost(ref_dir = None, results_dir = None):
    base = Path(__file__).resolve().parent.parent
    ref_dir = Path(ref_dir or base / "references")
    results_dir = Path(results_dir or base / "results")

    rows = {}

    if ref_dir.is_dir():
        for exp in sorted(ref_dir.iterdir()):
            train = exp / "train.txt"

            if exp.is_dir() and train.is_file():
                seen = sum(1 for line in open(train) if line.strip())
                rows.setdefault(exp.name, {})['immagini viste'] = seen

    # ultralytics scrive in results.csv una colonna 'time' in secondi cumulativi:
    # l'ultima riga e' il tempo totale di training. args.yaml da' il contesto
    # (batch, imgsz, device) ma NON la macchina - e i tempi di freeman e forrester
    # differiscono di 1.4x. L'hostname e' nel ledger, da benchmark.py in poi
    if results_dir.is_dir():
        for run in sorted(results_dir.iterdir()):
            csv = run / "results.csv"

            if not run.is_dir() or not csv.is_file():
                continue

            try:
                table = pd.read_csv(csv)
                table.columns = [c.strip() for c in table.columns]
            except Exception:
                continue

            if 'time' not in table.columns or table.empty:
                continue

            seconds = float(table['time'].iloc[-1])
            epochs = int(table['epoch'].iloc[-1]) if 'epoch' in table.columns else len(table)

            rows.setdefault(run.name, {})['epoche'] = epochs
            rows.setdefault(run.name, {})['minuti training'] = round(seconds / 60, 1)

    if not rows:
        return None

    table = pd.DataFrame.from_dict(rows, orient = 'index')

    # il costo per immagine per epoca: e' la grandezza che si trasferisce fra
    # macchine (6.05 ms su freeman, ~4.3 su forrester per yolo26n)
    if 'immagini viste' in table.columns and 'minuti training' in table.columns:
        table['ms / img / epoca'] = (
            table['minuti training'] * 60_000
            / (table['immagini viste'] * table['epoche'])
        ).round(2)

    return table

# media e deviazione per configurazione delle basi: l'esperimento sull'asse dei
# negativi. Riporta anche il rapporto fra la dispersione FRA configurazioni e
# quella ENTRO, cioe' se le configurazioni siano distinguibili
def base_axis(df, benchmark, columns = CURVE_COLUMNS):
    rows = {}

    for config, runs in BASE_CONFIGS.items():
        for column in columns:
            mean, sd, n = mean_sd(df, benchmark, column, runs)

            if n == 0:
                continue

            rows.setdefault(config, {})[f"{column} mean"] = mean
            rows.setdefault(config, {})[f"{column} sd"] = sd
            rows.setdefault(config, {})['n'] = n

    if not rows:
        return None

    table = pd.DataFrame.from_dict(rows, orient = 'index')

    ratios = {}

    for column in columns:
        means = table.get(f"{column} mean")
        sds = table.get(f"{column} sd")

        if means is None or sds is None or means.dropna().size < 2:
            continue

        within = sds.dropna().mean()
        between = means.dropna().std()
        ratios[column] = round(between / within, 2) if within and within > 0 else None

    return table.round(4), ratios

# le coppie a dati identici: quanto costa ripartire da zero invece di continuare
def paired(df, benchmark, column = MAP95):
    rows = {}

    for x, joint_run, replay_run in PAIRED:
        j = scalar(df, benchmark, column, run = joint_run)
        r = scalar(df, benchmark, column, run = replay_run)

        if j is None and r is None:
            continue

        rows[x] = {
            'joint (da zero, 100 ep)': j,
            'replay (dal checkpoint, 30 ep)': r,
            'replay - joint': (r - j) if (j is not None and r is not None) else None,
        }

    if not rows:
        return None

    table = pd.DataFrame.from_dict(rows, orient = 'index')
    table.index.name = 'quota del vecchio'

    return table.round(4)

# --- la matrice R del continual learning sequenziale ---------------------------
#
# Qui T = 4 e la matrice esiste davvero. R[i][j] e' la prestazione sul task j
# dopo il passo i; le metriche sono quelle di Lopez-Paz & Ranzato 2017 (ACC,
# BWT, FWT) e Chaudhry et al. 2018 (forgetting, intransigence), calcolate per
# catena e per seed, poi media +/- sd sui seed.
#
# Le run si trovano per NOME: seq{i}_<catena>_s{k} e' il checkpoint dopo il
# passo i. Il passo 1 e' condiviso fra le catene (seq1_s{k}), il joint e'
# seq_joint_s{k}, il modello mai allenato e' yolo26n.

SEQ_TASKS = [
    ('public',      'f1tenth_holdout'),
    ('squeaky',     'squeaky_holdout'),
    ('FIM_outside', 'FIM_outside_holdout'),
    ('icra',        'icra_holdout'),
]
SEQ_CHAINS = {'ft': 'seq{i}_ft_s{k}', 'ftr': 'seq{i}_ftr_s{k}'}
SEQ_STEP1 = 'seq1_s{k}'
SEQ_JOINT = 'seq_joint_s{k}'
SEQ_SEEDS = (1, 2, 3)

def seq_run(chain, step, seed):
    if step == 1:
        return SEQ_STEP1.format(k = seed)

    return SEQ_CHAINS[chain].format(i = step, k = seed)

# la matrice per una catena e un seed: righe = passi, colonne = task, piu' una
# colonna FIM_inside fuori matrice. Celle mancanti restano NaN
def r_matrix(df, chain, seed, column = MAP95):
    T = len(SEQ_TASKS)
    rows = {}

    for i in range(1, T + 1):
        run = seq_run(chain, i, seed)
        row = {}

        for name, bm in SEQ_TASKS:
            row[name] = scalar(df, bm, column, run = run)

        row[BM_UNSEEN] = scalar(df, BM_UNSEEN, column, run = run)
        rows[f"dopo passo {i}"] = row

    table = pd.DataFrame.from_dict(rows, orient = 'index')

    return table if table.notna().any().any() else None

# le cinque metriche da una matrice. Con T task:
#   ACC  = media_j R[T][j]
#   BWT  = media_{j<T} (R[T][j] - R[j][j])           negativo = oblio
#   FWT  = media_{j>1} (R[j-1][j] - b[j])            b = yolo26n
#   FGT  = media_{j<T} (max_{l<T} R[l][j] - R[T][j]) positivo = oblio
#   INT  = media_j (R*[j] - R[j][j])                 R* = seq_joint
def seq_metrics(df, chain, seed, column = MAP95):
    R = r_matrix(df, chain, seed, column)

    if R is None:
        return None

    names = [n for n, _ in SEQ_TASKS]
    T = len(names)
    M = R[names].to_numpy(dtype = float)     # M[i-1][j-1] = R[i][j]

    def ok(*v):
        return all(x is not None and x == x for x in v)

    acc = [M[T-1][j] for j in range(T) if ok(M[T-1][j])]
    bwt = [M[T-1][j] - M[j][j] for j in range(T-1) if ok(M[T-1][j], M[j][j])]
    fgt = [max(M[l][j] for l in range(T-1)) - M[T-1][j] for j in range(T-1)
           if ok(M[T-1][j], *[M[l][j] for l in range(T-1)])]

    baseline = [scalar(df, bm, column, run = BASELINE_RUN) for _, bm in SEQ_TASKS]
    fwt = [M[j-1][j] - baseline[j] for j in range(1, T) if ok(M[j-1][j], baseline[j])]

    joint = [scalar(df, bm, column, run = SEQ_JOINT.format(k = seed)) for _, bm in SEQ_TASKS]
    intr = [joint[j] - M[j][j] for j in range(T) if ok(joint[j], M[j][j])]

    mean = lambda v: sum(v) / len(v) if v else None

    return {
        'ACC': mean(acc), 'BWT': mean(bwt), 'FWT': mean(fwt),
        'forgetting': mean(fgt), 'intransigence': mean(intr),
        # la generalizzazione del modello finale, fuori matrice
        'generalizzazione': R[BM_UNSEEN].iloc[-1] if BM_UNSEEN in R else None,
    }

# media +/- sd sui seed, per catena. E' la tabella del capitolo sul sequenziale
def seq_summary(df, column = MAP95):
    rows = {}

    for chain in SEQ_CHAINS:
        per_seed = [m for m in (seq_metrics(df, chain, k, column) for k in SEQ_SEEDS) if m]

        if not per_seed:
            continue

        for key in per_seed[0]:
            values = [m[key] for m in per_seed if m[key] is not None]

            if not values:
                continue

            series = pd.Series(values)
            rows.setdefault(chain, {})[f"{key} mean"] = series.mean()
            rows.setdefault(chain, {})[f"{key} sd"] = series.std() if len(values) > 1 else None

        rows.setdefault(chain, {})['n seed'] = len(per_seed)

    if not rows:
        return None

    return pd.DataFrame.from_dict(rows, orient = 'index').round(4)

# la matrice media sui seed, per stamparla come in letteratura
def r_matrix_mean(df, chain, column = MAP95):
    mats = [m for m in (r_matrix(df, chain, k, column) for k in SEQ_SEEDS) if m is not None]

    if not mats:
        return None

    return (sum(mats) / len(mats)).round(4)


def main():
    full = load_ledger()

    # le sezioni qui sotto ragionano su best/last: i checkpoint intermedi si
    # leggono come traiettorie, con trajectory() piu' avanti
    df = final_only(full)
    n_epoch_rows = int(is_epoch_row(full).sum())

    if n_epoch_rows:
        print(f"[ledger: {len(full)} righe, di cui {n_epoch_rows} di checkpoint intermedi, escluse dalle tabelle]")

    for benchmark in sorted(df['benchmark'].unique()):
        print(f"\n=== {benchmark}")
        print(per_run(df, benchmark))

        if 'arm' not in df.columns:
            continue

        print("\nmAP50 per braccio x composizione")
        print(per_arm(df, benchmark, 'all/mAP50'))

        print("\nFAR per braccio x composizione")
        print(per_arm(df, benchmark, 'negatives/far'))

    if 'arm' not in df.columns:
        return

    present = set(df['benchmark'])

    # una tabella per ogni sonda del dominio vecchio: se le due concordano il
    # risultato e' robusto, se divergono l'oblio e' specifico del sotto-dominio
    # (e quello e' un risultato, non un problema)
    for bm_old in BM_OLD_ALL:
        if bm_old not in present:
            continue

        for tag in cl_tags(df):
            print(f"\n\n=== continual learning [{tag}]  —  oblio misurato su {bm_old}")
            print("forgetting: positivo = degradato sul dominio vecchio (mAP e FAR)")
            print("intransigence: positivo = non ha imparato il dominio nuovo quanto il joint")
            print()
            print(cl_metrics(df, tag, bm_old))
            print()
            print(transfer(df, tag).to_string())

    if BM_UNSEEN in present:
        print(f"\n\n=== generalizzazione su {BM_UNSEEN} (nessun braccio lo allena)")
        print(generalization(df))

    for benchmark in sorted(present):
        result = base_axis(df, benchmark)

        if result is None:
            continue

        table, ratios = result
        print(f"\n\n=== asse dei negativi su {benchmark}  (media +/- sd su 3 seed)")
        print("rapporto sd-fra / sd-entro: sotto 2 le configurazioni non sono distinguibili")
        print()
        print(table)
        print()
        print("rapporti:", ratios)

    for benchmark in sorted(present):
        table = paired(df, benchmark)

        if table is None:
            continue

        print(f"\n\n=== coppie a dati identici su {benchmark}: da zero contro dal checkpoint")
        print("stessa quota del vecchio + tutto ICRA; cambia solo il punto di partenza")
        print()
        print(table)

    for title, points in CURVES.items():
        printed = False

        for benchmark in sorted(present):
            table = curve(df, points, benchmark)

            if table is None:
                continue

            if not printed:
                print(f"\n\n=== curva: {title}")
                printed = True

            spot = knee(table)

            print(f"\n-- {benchmark}" + (f"   (ginocchio su {MAP}: x = {spot})" if spot is not None else ""))
            print(table)

    for benchmark in sorted(present):
        table = seeds(df, benchmark)

        if table is None:
            continue

        print(f"\n\n=== repliche di seed su {benchmark}")
        print("una differenza fra bracci piu' piccola della deviazione non e' affermabile")
        print()
        print(table)

    for benchmark in sorted(present):
        table = nested(df, benchmark)

        if table is None:
            continue

        print(f"\n\n=== varianza: base contro seed di fine-tuning su {benchmark}")
        print("se 'varia la base' e' piu' larga di 'varia il seed', le barre")
        print("d'errore basate sui soli seed sottostimano l'incertezza vera")
        print()
        print(table)

    # il tempo di inferenza, dal ledger (benchmark.py lo scrive dal meta del
    # dump): ms per immagine del solo forward, e la macchina su cui e' misurato
    if 'infer_ms_per_img' in df.columns:
        speed = (df.groupby(['run', 'host', 'device'], dropna = False)
                   [['infer_ms_per_img', 'wall_ms_per_img']]
                   .mean().round(2))

        if not speed.empty:
            print("\n\n=== tempo di inferenza (media sui benchmark)")
            print("infer = solo forward del modello; wall = con I/O, preprocess, NMS, dump")
            print()
            print(speed)

    # --- il sequenziale: matrice R e metriche a T = 4 --------------------------
    for chain in SEQ_CHAINS:
        R = r_matrix_mean(df, chain)

        if R is None:
            continue

        print(f"\n\n=== sequenziale [{chain}]: matrice R (media su seed), {MAP95}")
        print("riga = dopo il passo i; colonna = holdout del task j; FIM_inside fuori matrice")
        print()
        print(R)

    summary = seq_summary(df)

    if summary is not None:
        print("\n\n=== sequenziale: metriche di continual learning (T = 4)")
        print("BWT negativo = oblio; forgetting positivo = oblio; intransigence vs seq_joint")
        print()
        print(summary)

    budget = cost()

    if budget is not None:
        print("\n\n=== costo: immagini viste in training")
        print("senza questa colonna 'meglio retrain o fine-tuning' non ha risposta")
        print()
        print(budget)

if __name__ == "__main__":
    main()
