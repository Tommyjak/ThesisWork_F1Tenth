import random

CONF_OP = 0.25
IOU_OP = 0.5
IOU_START = 0.5
IOU_STOP = 0.95
IOU_STEP = 0.05

def get_iou_thresholds():
    thresholds = []
    value = IOU_START

    while value <= IOU_STOP + 1e-9:
        thresholds.append(round(value, 2))
        value += IOU_STEP

    return thresholds

IOU_THRESHOLDS = get_iou_thresholds()

def box_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = x2 - x1
    inter_h = y2 - y1

    if inter_w <= 0 or inter_h <= 0:
        return 0.0

    inter = inter_w * inter_h

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return inter / union

def pred_score_key(pred):
    return pred['score']

def sort_preds(preds, min_score = 0.0):
    selected = []

    for pred in preds:
        if pred['score'] >= min_score:
            selected.append(pred)

    selected.sort(key = pred_score_key, reverse = True)

    return selected

# greedy: ogni predizione, dalla piu' sicura alla meno, si prende il gt libero
# con cui si sovrappone di piu'. Un gt puo' essere assegnato una volta sola,
# quindi le detection duplicate diventano falsi positivi.
def match_preds(preds, gts, iou_thr):
    matched = []

    for _ in gts:
        matched.append(False)

    flags = []

    for pred in preds:
        best_iou = 0.0
        best_idx = -1

        for idx in range(len(gts)):
            if matched[idx]:
                continue

            value = box_iou(pred['box'], gts[idx]['box'])

            if value > best_iou:
                best_iou = value
                best_idx = idx

        if best_idx >= 0 and best_iou >= iou_thr:
            matched[best_idx] = True
            flags.append(True)
        else:
            flags.append(False)

    return flags

def count_gt(records):
    total = 0

    for record in records:
        total += len(record['gt'])

    return total

def count_status(records, status):
    total = 0

    for record in records:
        if record['status'] == status:
            total += 1

    return total

def entry_score_key(entry):
    return entry[0]

# il matching e' per immagine, la curva PR e' globale: le detection di tutte
# le immagini finiscono in una lista sola, riordinata per confidenza
def collect_entries(records, iou_thr):
    entries = []

    for record in records:
        preds = sort_preds(record['pred'])
        flags = match_preds(preds, record['gt'], iou_thr)

        for pred, flag in zip(preds, flags):
            entries.append((pred['score'], flag))

    entries.sort(key = entry_score_key, reverse = True)

    return entries

def compute_ap(entries, n_gt):
    if n_gt == 0:
        return None

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

    # inviluppo: la precisione diventa monotona decrescente guardando da destra
    for idx in range(len(precisions) - 2, -1, -1):
        if precisions[idx] < precisions[idx + 1]:
            precisions[idx] = precisions[idx + 1]

    ap = 0.0

    for idx in range(1, len(recalls)):
        ap += (recalls[idx] - recalls[idx - 1]) * precisions[idx]

    return ap

def compute_map(records):
    n_gt = count_gt(records)

    if n_gt == 0:
        return None, None

    aps = []

    for iou_thr in IOU_THRESHOLDS:
        entries = collect_entries(records, iou_thr)
        aps.append(compute_ap(entries, n_gt))

    map_50 = aps[0]
    map_50_95 = sum(aps) / len(aps)

    return map_50, map_50_95

def compute_operating(records, conf_op = CONF_OP, iou_thr = IOU_OP):
    tp = 0
    fp = 0
    fn = 0

    for record in records:
        preds = sort_preds(record['pred'], conf_op)
        flags = match_preds(preds, record['gt'], iou_thr)
        hits = 0

        for flag in flags:
            if flag:
                hits += 1

        tp += hits
        fp += len(flags) - hits
        fn += len(record['gt']) - hits

    precision = None
    recall = None
    f1 = None

    if tp + fp > 0:
        precision = tp / (tp + fp)
    if tp + fn > 0:
        recall = tp / (tp + fn)
    if precision is not None and recall is not None:
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            # ci sono gt e ci sono predizioni, ma nessuna azzeccata: F1 e' 0,
            # non indefinito
            f1 = 0.0

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

def compute_far(records, conf_op = CONF_OP):
    negatives = []

    for record in records:
        if record['status'] == 'negative':
            negatives.append(record)

    if not negatives:
        return {'far': None, 'fp_per_neg_image': None, 'n_neg': 0}

    dirty = 0
    boxes = 0

    for record in negatives:
        preds = sort_preds(record['pred'], conf_op)
        if preds:
            dirty += 1
        boxes += len(preds)

    return {
        'far': dirty / len(negatives),
        'fp_per_neg_image': boxes / len(negatives),
        'n_neg': len(negatives),
    }

def compute_metrics(records, conf_op = CONF_OP):
    map_50, map_50_95 = compute_map(records)
    operating = compute_operating(records, conf_op)
    far = compute_far(records, conf_op)

    metrics = {
        'n_images': len(records),
        'n_pos': count_status(records, 'positive'),
        'n_neg': count_status(records, 'negative'),
        'n_gt': count_gt(records),
        'mAP50': map_50,
        'mAP50_95': map_50_95,
    }

    metrics.update(operating)
    metrics['far'] = far['far']
    metrics['fp_per_neg_image'] = far['fp_per_neg_image']

    counts = group_counts(records, conf_op)

    metrics['f1_lo'], metrics['f1_hi'] = bootstrap_ci(counts, 'f1')
    metrics['far_lo'], metrics['far_hi'] = bootstrap_ci(counts, 'far')

    return metrics

N_BOOT = 1000

# il ricampionamento e' per sessione, non per frame: i frame consecutivi non
# sono indipendenti e gli intervalli verrebbero molto piu' stretti del vero
def group_counts(records, conf_op = CONF_OP, iou_thr = IOU_OP):
    counts = {}

    for record in records:
        group = record.get('group')
        bucket = counts.setdefault(group, {'tp': 0, 'fp': 0, 'fn': 0, 'neg': 0, 'dirty': 0})

        preds = sort_preds(record['pred'], conf_op)
        flags = match_preds(preds, record['gt'], iou_thr)
        hits = 0

        for flag in flags:
            if flag:
                hits += 1

        bucket['tp'] += hits
        bucket['fp'] += len(flags) - hits
        bucket['fn'] += len(record['gt']) - hits

        if record['status'] == 'negative':
            bucket['neg'] += 1

            if preds:
                bucket['dirty'] += 1

    return counts

def boot_value(totals, metric):
    tp = totals['tp']
    fp = totals['fp']
    fn = totals['fn']

    if metric == 'far':
        if totals['neg'] == 0:
            return None

        return totals['dirty'] / totals['neg']

    if metric == 'recall' and tp + fn > 0:
        return tp / (tp + fn)

    if metric == 'precision' and tp + fp > 0:
        return tp / (tp + fp)

    # stesse condizioni di compute_operating: senza ground truth o senza
    # predizioni l'F1 non e' definito, e l'intervallo deve dire lo stesso
    if metric == 'f1' and tp + fn > 0 and tp + fp > 0:
        return 2 * tp / (2 * tp + fp + fn)

    return None

def bootstrap_ci(counts, metric, n_boot = N_BOOT, seed = 42):
    groups = list(counts)

    if len(groups) < 2:
        return None, None

    rng = random.Random(seed)
    values = []

    for _ in range(n_boot):
        totals = {'tp': 0, 'fp': 0, 'fn': 0, 'neg': 0, 'dirty': 0}

        for _ in groups:
            bucket = counts[rng.choice(groups)]

            for key in totals:
                totals[key] += bucket[key]

        value = boot_value(totals, metric)

        if value is not None:
            values.append(value)

    if not values:
        return None, None

    values.sort()

    return values[int(0.025 * len(values))], values[int(0.975 * len(values))]
