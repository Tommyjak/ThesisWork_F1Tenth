import shutil
import random
from pathlib import Path

VALID_EXTS = {'.jpg', '.png', '.jpeg'}

HUAWEI_DS     = Path("/home/tm/Documents/universita/tirocinio_hipert/research/datasets/huawei-uci2020-dataset")
SQUEAKY_NEG   = Path("/home/tm/Documents/universita/tirocinio_hipert/research/datasets/squeaky_17-04-26/all_negatives")
OUTPUT_DIR    = Path("/home/tm/Documents/universita/tirocinio_hipert/research/datasets/miscellaneous_negatives")

def output_dirs_creation(base_dir):
    images_out = base_dir / "images"
    labels_out = base_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    return images_out, labels_out

def is_negative(image_path, labels_dir):
    label_path = labels_dir / f"{image_path.stem}.txt"
    if not label_path.exists():
        return True
    return label_path.stat().st_size == 0

def collect_negatives(images_dir, labels_dir):
    negatives = []
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in VALID_EXTS:
            continue
        if is_negative(img, labels_dir):
            negatives.append(img)
    return negatives

def sample_and_copy(negatives, n_samples, images_out, labels_out, tag, seed=42):
    if n_samples > len(negatives):
        print(f"[WARNING] {tag}: requested {n_samples} but only {len(negatives)} available. Using all.")
        n_samples = len(negatives)

    random.seed(seed)
    selected = random.sample(negatives, n_samples)

    for img in selected:
        new_name = f"{img.stem}_{tag}{img.suffix}"
        dst_img = images_out / new_name
        dst_lbl = labels_out / f"{img.stem}_{tag}.txt"

        if dst_img.exists():
            print(f"[SKIP] {new_name} already exists.")
            continue

        shutil.copy2(img, dst_img)
        dst_lbl.touch()  # empty label file

    print(f"[INFO] {tag}: copied {n_samples} negative samples.")

def clean_output_dir():
    if OUTPUT_DIR.exists():
        for item in OUTPUT_DIR.iterdir():
            if item.is_dir():
                shutil.rmtree(item)

def main():
    n_huawei  = 0
    n_squeaky = 500

    clean_output_dir()

    images_out, labels_out = output_dirs_creation(OUTPUT_DIR)

    # --- huawei negatives ---
    huawei_images = HUAWEI_DS / "images"
    huawei_labels = HUAWEI_DS / "labels"
    huawei_negatives = collect_negatives(huawei_images, huawei_labels)
    print(f"[INFO] huawei: found {len(huawei_negatives)} negatives.")
    sample_and_copy(huawei_negatives, n_huawei, images_out, labels_out, tag="huawei")

    # --- squeaky negatives ---
    squeaky_images = SQUEAKY_NEG / "images"
    squeaky_labels = SQUEAKY_NEG / "labels"
    squeaky_negatives = collect_negatives(squeaky_images, squeaky_labels)
    print(f"[INFO] squeaky: found {len(squeaky_negatives)} negatives.")
    sample_and_copy(squeaky_negatives, n_squeaky, images_out, labels_out, tag="squeaky")

    print(f"\n[DONE] Miscellaneous negatives saved at {OUTPUT_DIR}")
    print(f"[DONE] Total images: {len(list(images_out.iterdir()))}")

if __name__ == "__main__":
    main()