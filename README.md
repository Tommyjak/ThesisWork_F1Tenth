# Continual Learning Framework for Opponent Detection in Autonomous Car Races

Code for the MSc thesis in Artificial Intelligence Engineering (LM-32),
University of Modena and Reggio Emilia, "Enzo Ferrari" Department of Engineering.
Supervisor: Paolo Burgio. Co-supervisors: Alessandro Capotondi, Francesco Moretti.

Opponent racecar detection from the on-board camera of the
**RoboRacer (F1TENTH)** platform, using a single-class **YOLO26n** model
(`0: racecar`), studied through the lens of **continual learning**: data does
not arrive as a fixed dataset, but as successive recording sessions, each tied
to its own environment. The problem is therefore not just training a detector,
but deciding **how to integrate a new session** without the model forgetting the
previous ones.

This repository contains two contributions:

1. **An incremental dataset** of 16,607 images, collected in the field together
   with the HiPeRT team, designed to accept new sessions after its creation.
2. **A framework** that composes experiments from that dataset, trains them,
   evaluates them on separate benchmarks and records the results in an
   analysable form.

The data and the results are not here: they are too large (6.4 GB) and live
outside the repository. What lives here is the tool.

---

## The pipeline

```
datasets.yaml ─┐
experiments.yaml┼─► PREPROCESSING ─► references/<exp>/{train,val}.txt + config.yaml
                │   loading → composition → splitting → export
                │
                ├─► TRAINING  (train.py)          ─► results/<exp>/weights/best.pt
                │   Ultralytics YOLO, one run per experiment
                │
benchmarks.yaml ┴─► BENCHMARKING (benchmark.py)
                        │
                        ├─ inference ──► dump .jsonl + .meta.json   [GPU, once]
                        │                conf = 0.001, no operating threshold
                        │  ─────────────────────────────────────────────────────
                        ├─ slicing ────► subsets by status / area / session
                        ├─ metrics ────► mAP, FAR, P/R/F1 for every slice
                        └─ ledger ─────► one row per (model, benchmark)
                                               │
                                               ▼
                                   ANALYSIS (analysis.py, plot_*.py)
                                   tables, CL metrics, figures
```

The horizontal line is the framework's design principle: **above it sits
everything that costs GPU time and is done once; below it, everything that costs
seconds of CPU and can be redone as often as needed.** Adding a "small objects
only" slice today touches `slices.py` alone and applies retroactively to every
checkpoint already evaluated, with no re-inference and no retraining.

---

## Quick start

```bash
source .venv_utils/bin/activate

python3 training_supervised/train.py       # composes the splits and trains every experiment
python3 training_supervised/benchmark.py   # evaluates every model on every benchmark
cd training_supervised/evaluation && python3 analysis.py    # tables and metrics
cd training_supervised/evaluation && python3 new_plots.py   # figures into plots/
```

Paths are anchored to the location of the files, so `train.py` and
`benchmark.py` can be launched from any directory. The scripts in `evaluation/`
cannot: they read `ledger.jsonl` and write into `plots/` using relative paths,
so they must be run from inside `evaluation/`.

---

## Configuration

Four YAML files, no code to touch.

**`preprocessing/configs/datasets.yaml`** — the datasets available on disk.

```yaml
datasets:
  dataset1:
    name: "f1tenth.v2i.yolov11"
    root: "/absolute/path/to/research/datasets/f1tenth.v2i.yolov11"
    type: "subfolders"        # or "standard"
```

**`preprocessing/configs/experiments.yaml`** — which datasets to combine, in
what proportions, with which hyperparameters. A new experiment is a new key.

```yaml
experiments:
  exp1:
    name: "base_90_10_s1"
    model: 'yolo26n.pt'       # or init_from: "<another experiment>" for fine-tuning
    arm: "base"               # analytical label, used by analysis.py
    tag: "90_10_s1"
    hyp: {epochs: 100, patience: 100, seed: 1, save_period: 2}
    datasets:
      f1tenth.v2i.yolov11:    {pos_ratio: 1.0, neg_ratio: 0.1022}
      FIM_outside_base:       {pos_ratio: 1.0, neg_ratio: 0.1022}
    val_pct: 0.15
    seed: 42
```

`pos_ratio` and `neg_ratio` are fractions of the available pool (`1.0` = all of
it) and are the mechanism through which the experimental arms are expressed: the
replay budget, the share of historical data in a joint retrain, the fraction of
the new domain in a fine-tuning run.

**`benchmarking/configs/benchmarks.yaml`** — the evaluation-only sets. Same
schema plus `enabled`. They never enter training, so they have neither
`pos_ratio` nor `val_pct`.

```yaml
benchmarks:
  benchmark1:
    name: "icra_holdout"
    root: "/absolute/path/.../icra_test_camera/icra_holdout"
    type: "subfolders"
    enabled: true
```

The three parsers accept an environment-variable override
(`DS_YAML`, `EXP_YAML`, `BM_YAML`), so an alternative campaign runs on the same
scripts without editing the main files — this is the mechanism behind the
`tmp_*.yaml` files of the sequential campaign.

---

## Data convention

Ultralytics layout: `images/` and `labels/`, one `.txt` per image with the same
name. YOLO derives the label path by replacing `/images/` with `/labels/`, so
the preprocessing stage only ever manipulates image paths.

- **positive** — label present and non-empty (`class cx cy w h`, normalised)
- **negative** — label missing or empty: an image with no opponent

Negatives make up almost 60% of the dataset and are first-class data, not
discards: they teach the model **not** to detect, and without them the FAR would
not exist — the metric that matters most in a safety-critical context.

| `type`       | expected structure |
|--------------|--------------------|
| `standard`   | `<root>/images` + `<root>/labels` (single session) |
| `subfolders` | `<root>/<session>/images` + `labels`, for every subfolder |

For a different format: a class in `core/loaders/` extending `Loader` and
implementing `find_samples()`, plus a branch in `loading.py`.

---

## Preprocessing

Two dataclasses carry everything (`core/`):

```python
@dataclass
class ImageSample:
    status: LblStatus          # positive | negative
    ds_name: str               # source dataset
    img_path: Path
    lbl_path: Path | None
    split: Split | None
    group: str | None          # recording sub-session

@dataclass
class Dataset:
    name: str
    root: Path
    pos_samples: list[ImageSample]
    neg_samples: list[ImageSample]
```

The `group` field is what makes block bootstrapping possible at analysis time:
images from the same session are strongly correlated, and resampling frames
instead of sessions would produce confidence intervals that are falsely narrow.

The four stages, orchestrated by `pipeline.py`:

- **`loading.py`** picks a loader by `type` and returns whole `Dataset` objects
  with the samples already separated. This is where format differences
  disappear: downstream, nothing knows how the files were organised.
- **`composition.py`** selects the experiment's datasets and trims the pools
  according to `pos_ratio` / `neg_ratio`.
- **`splitting.py`** assigns train/val **to each pool separately** before
  merging them, so both splits preserve the same positive/negative proportions
  and the same per-dataset composition. Each experiment uses its own
  `random.Random(seed)`: the splits are reproducible.
- **`export.py`** writes `train.txt`, `val.txt` (one absolute path per line) and
  the `config.yaml` Ultralytics expects. The `.txt` files are at once the
  interface towards YOLO and an inspectable record of how the experiment was
  composed.

Training uses the defaults in `train.py` (100 epochs, `imgsz=640` matched to the
on-vehicle inference resolution, AdamW with `lr0=0.001`, mosaic enabled,
`device=0`), overridable per experiment through the `hyp` block.

---

## Benchmarking

Testing is kept separate from training and validation, so that the images the
performance is measured on are under explicit control.

**Inference** (`prediction.py`) — one pass per (model, benchmark) pair, at
`conf=0.001`, `iou=0.7`, `imgsz=640`. Every image yields a record that carries
its own ground truth, which makes metric computation self-sufficient: it never
has to read the `.txt` files again.

```json
{"image": ".../frame_000.jpg", "group": "test_1_03_09_2026", "status": "positive",
 "w": 1280, "h": 720,
 "pred": [{"box": [690.61, 89.68, 1271.66, 600.39], "score": 0.86274}],
 "gt":   [{"box": [688.54, 73.01, 1265.64, 602.17]}]}
```

Next to each dump sits a `.meta.json` describing the conditions under which it
was produced: `meta_version`, the weights path and their **SHA-256**, the number
of images, the inference parameters, host, GPU, torch version and per-image
timings. The hash is the cache key: if a dump exists and is consistent,
inference is skipped. Host and device exist to prevent comparing measurements
taken on different machines.

The very low `conf=0.001` threshold is deliberate: it preserves the entire
precision-recall curve, which Average Precision requires, and makes the dump
reusable at any future threshold.

**Slicing** (`slices.py`) — partitions records already on disk, at two levels:

- *per image* — positive/negative, session of origin
- *per object* — bounding-box area band (COCO thresholds 32² and 96² on absolute
  area), which acts as a proxy for the opponent's distance

The second level is not a filter but a rewrite: boxes must be removed
symmetrically from `pred` **and** `gt`, otherwise predictions on the excluded
objects would turn into false positives and the FAR would blow up.

**Metrics** (`metrics.py`) — mAP50 and mAP50-95 over the full PR curve, with no
confidence threshold; precision, recall, TP/FP/FN and F1 at the operating point
`CONF_OP=0.25` with `IOU_OP=0.5`; FAR over negative images only, defined as the
fraction of negative images containing at least one prediction above threshold.
Confidence intervals come from a block bootstrap that resamples **sessions**,
not frames.

**Ledger** (`ledger.py`) — one JSONL row per (model, benchmark) pair, with
columns named `<slice>/<metric>`. JSONL rather than CSV because the number of
slices changes over time. Writing is an upsert, so re-running the benchmark is
idempotent. With `save_period` enabled the ledger also holds one row per
intermediate checkpoint, carrying `epoch` and `train_s`: these are the
trajectories used in the figures.

---

## Analysis

`evaluation/analysis.py` reads the ledger and produces the tables: paired
comparisons between arms, dispersion across seeds, training-time cost, and the
continual learning metrics — accuracy matrix R, ACC, BWT, FWT (Lopez-Paz &
Ranzato, 2017), forgetting and intransigence (Chaudhry et al., 2018).

`new_plots.py`, `plot_conf_smooth.py`, `plot_base_smooth.py` and `plot_vari.py`
produce the figures in `evaluation/plots/`: per-model trajectories, comparisons
between arms on each benchmark, and the Pareto curve of training time against
mAP.

### Campaigns

A campaign is a self-contained set of ledger, dumps, references and training
results. `use_campaign.sh` moves four symlinks to select the active one, copying
nothing:

```bash
./training_supervised/use_campaign.sh              # show which one is active
./training_supervised/use_campaign.sh sequential   # switch to the sequential campaign
```

The code stays unaware: it keeps reading `results/`, `references/`,
`evaluation/ledger.jsonl` and `evaluation/predictions` as always.

---

## Layout

```text
utils/
├── vid2img/                       # MCAP → MP4@30fps → frames@10fps
├── requirements.txt
└── training_supervised/
    ├── train.py                   # preprocessing + training, iterated over experiments
    ├── benchmark.py               # enumerates checkpoints and evaluates them
    ├── use_campaign.sh            # selects the active campaign
    ├── core/                      # ImageSample, Dataset, loaders
    ├── preprocessing/
    │   ├── pipeline.py  loading.py  composition.py  splitting.py  export.py
    │   └── configs/               # datasets.yaml, experiments.yaml, parsers
    ├── benchmarking/
    │   ├── pipeline.py  prediction.py  slices.py  metrics.py  ledger.py
    │   ├── configs/               # benchmarks.yaml, parser
    │   └── senso.md               # design notes on the pipeline
    ├── evaluation/
    │   ├── analysis.py            # tables and continual learning metrics
    │   ├── ratios.py              # positive/negative counts per dataset
    │   ├── new_plots.py  plot_conf_smooth.py  plot_base_smooth.py  plot_vari.py
    │   └── plots/                 # generated figures (contents not versioned)
    ├── models/                    # starting weights (not versioned)
    ├── references/ → campaigns/   # generated splits   (symlink, not versioned)
    ├── results/    → campaigns/   # training output    (symlink, not versioned)
    └── campaigns/                 # ledgers, dumps, weights, references (not versioned)
```

---

## The dataset

16,607 images, 6,718 positive and 9,889 negative, 83% of them recorded directly
with the team. The acquisition chain is
**ROS2 bag (MCAP) → MP4 at 30 fps → frames at 10 fps → annotation in CVAT**:
sampling at 10 fps keeps continuity between consecutive frames without producing
useless duplicates, and the intermediate MP4 step exists precisely so that the
chain can also be entered from a plain external video recording.

| dataset | images | positive | negative |
|---|---:|---:|---:|
| f1tenth.v2i.yolov11 | 2,283 | 2,283 | 0 |
| huawei-uci2020-dataset | 514 | 60 | 454 |
| squeaky_17-04-26 | 3,270 | 850 | 2,420 |
| objects_negatives_15-05-26 | 1,165 | 0 | 1,165 |
| FIM_outside | 3,378 | 1,932 | 1,446 |
| FIM_inside | 2,441 | 894 | 1,547 |
| icra_test_camera | 3,556 | 699 | 2,857 |
| **total** | **16,607** | **6,718** | **9,889** |

The heterogeneity is intentional: indoor and outdoor, different lighting, points
of view and backgrounds. `objects_negatives` contains boxes and objects that can
be mistaken for a car, and is the hard-negative pool.

---

## What was measured

The framework was used to answer three questions about how to integrate a new
session, comparing **joint retrain** (retraining from scratch on everything),
**naive fine-tuning** (new data only) and **fine-tuning with experience replay**
(new data plus a share of the historical set), each at several data budgets and
repeated over three seeds. The three benchmarks measure, respectively,
forgetting of the previous domain, generalisation to an environment never
trained on, and specialisation on the newly acquired one.

The conclusion is that no approach is best in every situation: replay gives the
best trade-off between retention and adaptation, joint retrain the highest
absolute values, and on specialisation alone the configurations are nearly
indistinguishable. Once training time enters the picture the balance shifts:
joint retrain can cost 27 times as much as fine-tuning for an almost identical
mAP50-95.

The complete figures, tables and discussion are in the thesis.

---

## Known limitations

- The train/val split is random within pools, not temporal: adjacent frames from
  the same video can land in both, which may overestimate validation
  performance. A session-level split is the natural next step.
- Paths in `datasets.yaml` and `benchmarks.yaml` are absolute and must be
  updated on another machine.
- `references/` and `results/` are regenerated on every run: do not keep
  anything there.
- The experimental campaigns (6.4 GB of ledgers, dumps and weights) are not in
  the repository. Without them the analysis scripts have no data to read.
