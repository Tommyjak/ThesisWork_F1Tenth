AREA_BANDS = {
    'small': (0.0, 32.0 * 32.0),
    'medium': (32.0 * 32.0, 96.0 * 96.0),
    'large': (96.0 * 96.0, float('inf')),
}

def box_area(box):
    width = box[2] - box[0]
    height = box[3] - box[1]

    if width <= 0 or height <= 0:
        return 0.0

    return width * height

def is_in_band(box, band):
    min_area, max_area = band
    area = box_area(box)

    return area >= min_area and area < max_area

def new_slice(name, status = None, group = None, band = None):
    return {
        'name': name,
        'status': status,
        'group': group,
        'band': band,
    }

def get_groups(records):
    groups = []

    for record in records:
        group = record.get('group')

        if group is None:
            continue

        if group not in groups:
            groups.append(group)

    return sorted(groups)

def build_slices(records):
    slices = []

    slices.append(new_slice('all'))
    slices.append(new_slice('positives', status = 'positive'))
    slices.append(new_slice('negatives', status = 'negative'))

    for band_name in AREA_BANDS:
        slices.append(new_slice(f"area_{band_name}", band = AREA_BANDS[band_name]))

    for group in get_groups(records):
        slices.append(new_slice(f"group_{group}", group = group))

    return slices

def filter_by_status(records, status):
    selected = []

    for record in records:
        if record['status'] == status:
            selected.append(record)

    return selected

def filter_by_group(records, group):
    selected = []

    for record in records:
        if record.get('group') == group:
            selected.append(record)

    return selected

def filter_by_band(records, band):
    selected = []

    for record in records:
        kept_gt = []

        for gt in record['gt']:
            if is_in_band(gt['box'], band):
                kept_gt.append(gt)

        kept_pred = []

        for pred in record['pred']:
            if is_in_band(pred['box'], band):
                kept_pred.append(pred)

        trimmed = dict(record)
        trimmed['gt'] = kept_gt
        trimmed['pred'] = kept_pred
        selected.append(trimmed)

    return selected

def apply_slice(records, spec):
    selected = records

    if spec['status'] is not None:
        selected = filter_by_status(selected, spec['status'])

    if spec['group'] is not None:
        selected = filter_by_group(selected, spec['group'])

    if spec['band'] is not None:
        selected = filter_by_band(selected, spec['band'])

    return selected
