import sys
from pathlib import Path
from ultralytics import YOLO

sys.path.append(str(Path(__file__).parent))

from preprocessing.pipeline import run_preprocessing
import preprocessing.configs.exp_yaml_parser as eyp 

BASE_DIR = Path(__file__).resolve().parent
REF_DIR = BASE_DIR / "references"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = BASE_DIR / "models/yolo26n.pt"

TRAIN_DEFAULTS = dict(
    # --- core ---
    epochs = 100,
    patience = 20,          # early stopping: stop if no improvement for 20 epochs
    imgsz = 640,            # allineare alla risoluzione di inferenza sul veicolo
    batch = 16,             # fisso: -1 dipende dalla VRAM libera -> run non riproducibili
    
    # --- optimizer ---
    optimizer = 'AdamW',    # more stable than SGD for fine-tuning
    lr0 = 0.001,            # lower than default (0.01) since we're fine-tuning
    lrf = 0.01,             # final lr = lr0 * lrf -> 0.00001
    warmup_epochs = 3,      # riscalato sotto in base a epochs
    weight_decay = 0.0005,

    # --- loss weights ---
    cls = 0.5,              # default: con 1 classe il cls score E' la confidence
    
    # --- augmentation ---
    hsv_h = 0.015,          # hue shift (keep low, indoor lighting is consistent)
    hsv_s = 0.5,            # saturation
    hsv_v = 0.5,            # brightness/value (handles shadow variation)
    fliplr = 0.5,           # horizontal flip (opponent can come from either side)
    flipud = 0.0,           # vertical flip (disable: cars don't appear upside down)
    mosaic = 1.0,           # simulates multiple opponents/distances
    mixup = 0.0,            # off: crea auto semitrasparenti -> falsi positivi
    scale = 0.5,            # scale jitter (simulates distance variation)
    translate = 0.1,
    degrees = 5.0,          # small rotation (camera is roughly fixed on car)
    perspective = 0.0001,   # subtle perspective warp
    
    # --- evaluation ---
    iou = 0.7,              # NMS IoU. NON e' la soglia della mAP (fissa a 0.5:0.95)
    
    # --- hardware ---
    device = 0,
    workers = 8,

    # --- reproducibility ---
    seed = 42,
    exist_ok = False,       # errore invece di sovrascrivere una run gia' fatta
)

def build_args(hyp=None):
    args = {**TRAIN_DEFAULTS, **(hyp or {})}

    args['warmup_epochs'] = min(args['warmup_epochs'], max(1, args['epochs'] // 5))

    args['close_mosaic'] = min(10, args['epochs'] // 2)
    return args

def train_model(config, run_name, weights, hyp=None):
    print(f"[{run_name}] loading model {weights.name}...")
    model = YOLO(weights)
    print(f"[{run_name}] starting training...")

    results = model.train(
        data = str(config),
        project = str(RESULTS_DIR),
        name = run_name,
        **build_args(hyp)
    )

    print(f"[{run_name}] done -> {results.save_dir}")

    return results

def get_init_weights(run_name, models, meta):
    init_from = meta[run_name]['init_from']

    if init_from is None:
        weights = MODELS_DIR / models[run_name]

        if not weights.is_file():
            raise FileNotFoundError(f"[{run_name}] modello base non trovato: {weights}")

        return weights

    weights = RESULTS_DIR / init_from / "weights" / "best.pt"

    if not weights.is_file():
        raise FileNotFoundError(f"[{run_name}] init_from '{init_from}' non ancora allenato: {weights}")

    return weights

def run_training_experiments():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    meta = eyp.get_exp_meta()

    experiments = sorted(
        (e for e in REF_DIR.iterdir() if e.is_dir()),
        key = lambda e: (meta[e.name]['init_from'] is not None, e.name)
    )

    if not experiments:
        raise RuntimeError(f"Nessun esperimento in {REF_DIR}")

    models = dict(zip(eyp.get_exp_names(), eyp.get_exp_models()))
    done, skipped = [], []

    for exp in experiments:
        run_name = exp.name

        if (RESULTS_DIR / run_name).exists():
            print(f"[{run_name}] gia' presente in results/, salto")
            skipped.append(run_name)
            continue

        weights = get_init_weights(run_name, models, meta)

        train_model(exp / "config.yaml", run_name, weights, meta[run_name]['hyp'])
        done.append(run_name)

    print(f"\nAllenati: {len(done)}  |  Saltati: {len(skipped)}")

    for name in skipped:
        print(f"  saltato: {name}  (cancella results/{name} per rifarlo)")

def main():
    run_preprocessing(REF_DIR)
    run_training_experiments()

if __name__ == "__main__":
    main()