import shutil
from pathlib import Path

VALID_EXTS = {
    '.jpg',
    '.png',
    '.jpeg'
}

def output_dirs_creation(base_dir):
    images_out = base_dir / "complete_labels" / "images"
    labels_out = base_dir / "complete_labels" / "labels"
    negatives_out = base_dir / "all_negatives" / "images"
    negatives_labels_out = base_dir / "all_negatives" / "labels"

    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    negatives_out.mkdir(parents=True, exist_ok=True)
    negatives_labels_out.mkdir(parents=True, exist_ok=True)

    return images_out, labels_out, negatives_out, negatives_labels_out

def find_image_for_label(images_src, stem):
    for ext in VALID_EXTS:
        candidate = images_src / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None

def is_positive_label(label_path):
    """A label is positive if it exists and is non-empty."""
    return label_path.exists() and label_path.stat().st_size > 0

def check_link_and_create_files(folder, images_src, labels_src, images_out, labels_out, negatives_out, negatives_labels_out):
    for img_path in sorted(images_src.iterdir()):
        if img_path.suffix.lower() not in VALID_EXTS:
            continue

        stem = img_path.stem
        label_path = labels_src / f"{stem}.txt"
        new_name_img = f"{stem}_{folder.name}{img_path.suffix}"
        new_name_lbl = f"{stem}_{folder.name}.txt"

        if is_positive_label(label_path):
            # --- positive sample ---
            dst_img = images_out / new_name_img
            dst_lbl = labels_out / new_name_lbl

            if dst_lbl.exists():
                print(f"[SKIP] Positive already exists: {new_name_lbl}")
                continue

            shutil.copy2(img_path, dst_img)
            shutil.copy2(label_path, dst_lbl)

        else:
            # --- negative sample (missing or empty label) ---
            dst_img = negatives_out / new_name_img
            dst_lbl = negatives_labels_out / new_name_lbl

            if dst_img.exists():
                print(f"[SKIP] Negative already exists: {new_name_img}")
                continue

            shutil.copy2(img_path, dst_img)
            (negatives_labels_out / new_name_lbl).touch()  # empty label file


def labelled_folder_creation(base_dir, base_bag_name):
    images_out, labels_out, negatives_out, negatives_labels_out = output_dirs_creation(base_dir)

    source_dirs = sorted(base_dir.glob(f"{base_bag_name}*"))

    pos_total, neg_total = 0, 0

    for folder in source_dirs:
        images_src = folder / "images"
        labels_src = folder / "labels"

        if not images_src.exists():
            print(f"[WARNING] No images/ found in {folder.name}, skipping.")
            continue

        if not labels_src.exists():
            print(f"[WARNING] No labels/ found in {folder.name}, treating all as negatives.")

        pos_before = len(list(images_out.iterdir()))
        neg_before = len(list(negatives_out.iterdir()))

        check_link_and_create_files(
            folder, images_src, labels_src,
            images_out, labels_out,
            negatives_out, negatives_labels_out
        )

        pos_after = len(list(images_out.iterdir()))
        neg_after = len(list(negatives_out.iterdir()))

        pos_added = pos_after - pos_before
        neg_added = neg_after - neg_before
        pos_total += pos_added
        neg_total += neg_added

        print(f"[INFO] {folder.name}: +{pos_added} positives, +{neg_added} negatives")

    print(f"\n[DONE] Total positives: {pos_total} -> complete_labels/")
    print(f"[DONE] Total negatives: {neg_total} -> all_negatives/")

def main():
    base_dir = Path("/home/tm/Documents/universita/tirocinio_hipert/research/datasets/squeaky_17-04-26")
    base_bag_name = "2Macchine_"

    print(f"[INFO] Processing bags in: {base_dir}")
    print(f"[INFO] Looking for folders matching: {base_bag_name}*")
    print("-" * 50)

    labelled_folder_creation(base_dir, base_bag_name)

if __name__ == "__main__":
    main()