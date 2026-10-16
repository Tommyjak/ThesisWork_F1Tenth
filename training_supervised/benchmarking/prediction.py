import hashlib
import json
import platform
import time
from pathlib import Path

import torch
from ultralytics import YOLO

CONF = 0.001

# versione del formato di meta.json. Alzarla invalida i dump esistenti e forza
# la re-inferenza: e' voluto, perche' un dump senza tempi e macchina non basta
# piu' - il tempo e' una metrica e freeman e forrester differiscono di 1.4x
META_VERSION = 2
IOU_NMS = 0.7
IMG_SZ = 640
MAX_DET = 300
CHUNK_SZ = 1024 * 1024

# la macchina su cui l'inferenza e' girata. Senza, due ledger prodotti su GPU
# diverse sono indistinguibili e i tempi non sono confrontabili
def describe_host(device = None):
    cuda = torch.cuda.is_available() and (device is None or str(device) != 'cpu')

    return {
        'host': platform.node(),
        'device': torch.cuda.get_device_name(0) if cuda else 'cpu',
        'torch': torch.__version__,
    }

def write_meta(meta_path, weights, sha, benchmark_name, n_samples, timing = None, device = None):
    meta = {
        'meta_version': META_VERSION,
        'weights': str(weights),
        'weights_sha': sha,
        'benchmark': benchmark_name,
        'n_images': n_samples,
        'conf': CONF,
        'iou': IOU_NMS,
        'imgsz': IMG_SZ,
    }

    meta.update(describe_host(device))

    if timing is not None:
        meta.update(timing)

    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

def read_gt(lbl_path, w, h):
    boxes = []

    if lbl_path is None:
        return boxes

    if not Path(lbl_path).is_file():
        return boxes

    with open(lbl_path, 'r') as f:
        for line in f:
            parts = line.split()

            if len(parts) < 5:
                continue

            cx = float(parts[1])
            cy = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])

            x1 = (cx - (bw / 2)) * w
            y1 = (cy - (bh / 2)) * h
            x2 = (cx + (bw / 2)) * w
            y2 = (cy + (bh / 2)) * h

            boxes.append({
                'box': [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
            })

    return boxes

def read_pred(result):
    boxes = []

    xyxy = result.boxes.xyxy.tolist()
    scores = result.boxes.conf.tolist()

    for box, score in zip(xyxy, scores):
        rounded = []

        for value in box:
            rounded.append(round(value, 2))

        boxes.append({'box': rounded, 'score': round(score, 5)})

    return boxes

def write_sources(sources_path, samples):
    sample_by_path = {}

    with open(sources_path, 'w') as f:
        for sample in samples:
            key = str(Path(sample.img_path))
            sample_by_path[key] = sample
            f.write(key + '\n')

    return sample_by_path

def is_dump_valid(out_path, meta_path, sha, n_samples):
    if not out_path.is_file():
        return False

    if not meta_path.is_file():
        return False

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    # un dump di una versione precedente non ha tempi ne' macchina: si rifa'
    if meta.get('meta_version') != META_VERSION:
        return False

    if meta.get('weights_sha') != sha:
        return False

    if meta.get('n_images') != n_samples:
        return False

    if meta.get('conf') != CONF:
        return False

    if meta.get('iou') != IOU_NMS:
        return False

    if meta.get('imgsz') != IMG_SZ:
        return False

    return True

def weights_sha(weights):
    hasher = hashlib.sha256()

    with open(weights, 'rb') as f:
        while True:
            chunk = f.read(CHUNK_SZ)

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()

def sample_sort_key(sample):
    return str(sample.img_path)

def get_samples(benchmark):
    samples = benchmark.get_positive_samples() + benchmark.get_negative_samples()
    samples.sort(key = sample_sort_key)

    return samples

def predict_benchmark(weights, benchmark, out_path, device = None):
    weights = Path(weights)
    out_path = Path(out_path)
    meta_path = out_path.with_suffix('.meta.json')
    sources_path = out_path.with_suffix('.sources.txt')

    samples = get_samples(benchmark)
    sha = weights_sha(weights)

    if is_dump_valid(out_path, meta_path, sha, len(samples)):
        print(f"[{out_path.name}] dump gia' presente, salto")
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)

    sample_by_path = write_sources(sources_path, samples)

    model = YOLO(weights)

    results = model.predict(
        source = str(sources_path),
        stream = True,
        conf = CONF,
        iou = IOU_NMS,
        imgsz = IMG_SZ,
        max_det = MAX_DET,
        half = False,
        device = device,
        verbose = False,
    )

    written = 0

    # due orologi. Quello di ultralytics (result.speed) misura solo il forward
    # del modello, per immagine, e quindi si confronta con la latenza a bordo.
    # Quello a muro include lettura del file, preprocess, NMS e scrittura del
    # dump: e' il costo reale del benchmark, non del modello
    model_ms = 0.0
    wall_start = time.perf_counter()

    with open(out_path, 'w') as f:
        for result in results:
            key = str(Path(result.path))
            sample = sample_by_path[key]

            speed = getattr(result, 'speed', None) or {}
            model_ms += float(speed.get('inference') or 0.0)

            h, w = result.orig_shape

            record = {
                'image': str(sample.img_path),
                'group': sample.group,
                'status': sample.status.value,
                'w': w,
                'h': h,
                'pred': read_pred(result),
                'gt': read_gt(sample.lbl_path, w, h),
            }

            f.write(json.dumps(record) + '\n')

            written += 1

    wall_s = time.perf_counter() - wall_start

    if written != len(samples):
        raise RuntimeError(f"{out_path.name}: {written} record scritti su {len(samples)} attesi")

    timing = {
        'infer_ms_per_img': round(model_ms / written, 3) if written else None,
        'wall_ms_per_img': round(1000.0 * wall_s / written, 3) if written else None,
        'wall_s': round(wall_s, 2),
    }

    write_meta(meta_path, weights, sha, benchmark.name, len(samples), timing, device)

    return out_path