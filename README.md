# utils — detection dell'auto avversaria (F1Tenth / RoboRacer)

Training e inference di un modello **YOLO** (Ultralytics) per rilevare l'auto
avversaria, pensato per l'esecuzione su edge device. Una sola classe:
`0: racecar`. Tesi in AIE @ Unimore.

## Uso

```bash
source .venv_utils/bin/activate
python3 training_supervised/train.py        # rigenera gli split e allena
python3 training_supervised/inference.py    # detection live da webcam
```

`train.py` allena un modello per ogni esperimento definito in
`experiments.yaml`. I path sono ancorati alla posizione dei file: la directory da
cui lanci è indifferente.

## Configurare

I due soli file da toccare stanno in `training_supervised/preprocessing/configs/`.

**`datasets.yaml`** — i dataset disponibili. I dati veri non stanno nel repo:
vivono in `research/datasets/`, cartella sorella di `utils/`.

```yaml
datasets:
  dataset1:
    name: "f1tenth.v2i.yolov11"
    root: "/percorso/assoluto/a/research/datasets/f1tenth.v2i.yolov11"
    type: "subfolders"        # oppure "standard"
```

**`experiments.yaml`** — quali dataset combinare e in che proporzioni. I nomi
sotto `datasets:` devono corrispondere ai `name` di `datasets.yaml`; un nuovo
esperimento è una nuova chiave `exp2`.

```yaml
experiments:
  exp1:
    name: "full_with_negatives"     # dà il nome alla cartella di output
    datasets:
      f1tenth.v2i.yolov11:    {pos_ratio: 1.0, neg_ratio: 1.0}
      huawei-uci2020-dataset: {pos_ratio: 1.0, neg_ratio: 0.5}
    train_pct: 0.7
    val_pct: 0.15
    test_pct: 0.15
    seed: 42
```

## Convenzione dati

Layout Ultralytics: `images/` e `labels/`, un `.txt` per immagine dallo stesso
nome. YOLO ricava la label sostituendo `/images/` con `/labels/`, quindi il
preprocessing manipola solo path di immagini.

- **positivo** — label esistente e non vuota (`class cx cy w h` normalizzati)
- **negativo** — label mancante o vuota: immagine senza avversario

I negativi riducono i falsi positivi e sono dati di pari dignità, non scarti.

## Come funziona

```
research/datasets/<ds>/  +  datasets.yaml  +  experiments.yaml
                          │
                          │   loading -> composition -> splitting -> export
                          ▼
        training_supervised/references/<esperimento>/
            train.txt   val.txt   test.txt   config.yaml
                          │
                          ▼
                YOLO.train(data=config.yaml)
```

**loading.py** — legge `datasets.yaml` e sceglie un loader per `type`. Ogni
loader scandisce il disco e ritorna un `Dataset` con i sample già separati in
positivi e negativi. È qui che le differenze di formato spariscono: a valle
nessuno sa più com'erano organizzati i file.

| `type`       | struttura attesa                                        |
|--------------|---------------------------------------------------------|
| `standard`   | `<root>/images` + `<root>/labels`                       |
| `subfolders` | `<root>/<sub>/images` + `<root>/<sub>/labels`, per ogni sottocartella |

Per un formato diverso: una classe in `loaders/` che estende `Loader` e
implementa `find_samples()`.

**composition.py** — seleziona i dataset dell'esperimento e limita i pool secondo
`pos_ratio` / `neg_ratio` (`1.0` = tutto).

**splitting.py** — assegna train/val/test **a ogni pool separatamente**, prima di
unirli. Così i tre split mantengono le stesse proporzioni positivi/negativi e la
stessa composizione per dataset; splittare sull'aggregato darebbe una varianza
sensibile sui pool piccoli. Ogni esperimento usa un `random.Random(seed)`, quindi
gli split sono riproducibili.

**export.py** — scrive una cartella per esperimento con i tre `.txt` (un path
assoluto per riga, positivi e negativi mescolati) e il `config.yaml` che li
indica a Ultralytics. I `.txt` sono l'interfaccia verso YOLO e insieme la traccia
ispezionabile di come è stato composto l'esperimento.

## Struttura

```text
utils/
├── vid2img/                      # video .mp4 -> frame .jpg (passo manuale)
├── training_supervised/
│   ├── train.py                  # entrypoint: preprocessing + training
│   ├── inference.py              # detection live da webcam
│   ├── models/                   # pesi YOLO, base e addestrati
│   ├── references/               # OUTPUT rigenerato a ogni run (gitignored)
│   └── preprocessing/
│       ├── pipeline.py           # orchestrazione dei quattro stadi
│       ├── loading.py  composition.py  splitting.py  export.py
│       ├── configs/              # datasets.yaml, experiments.yaml, parser
│       ├── core/                 # ImageSample, Dataset
│       └── loaders/              # un loader per tipo di dataset
└── requirements.txt
```

## Dati grezzi

Passo manuale, una tantum, prima di registrare un dataset in `datasets.yaml`:

```bash
./vid2img/run.sh <video.mp4>          # estrae i frame a ~10 FPS
./vid2img/run_multiple.sh <cartella>  # tutti gli .mp4 di una cartella
```

Le immagini estratte vanno poi etichettate con un tool esterno, non coperto da
questo repo.

## Note

- `references/` è rigenerata a ogni run: non metterci file da conservare.
- `train.py` usa `device='cpu'`; per un training reale impostare `0` o `'cuda'`.
- I path in `datasets.yaml` sono assoluti: vanno aggiornati su un'altra macchina.
