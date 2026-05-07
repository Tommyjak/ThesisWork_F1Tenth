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

    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    return images_out, labels_out

def find_image_for_label(images_src, stem):
    for ext in VALID_EXTS:
        candidate = images_src / f"{stem}{ext}"
        if candidate.exists():
            return candidate
        
    return None

def check_link_and_create_files(folder, images_src, labels_src, images_out, labels_out):
    for label in sorted(labels_src.iterdir()):
        if not label.is_file():
            continue

        stem = label.stem
        suffix = label.suffix

        new_name = f"{stem}_{folder.name}{suffix}"

        lbl_dst = labels_out / new_name

        if lbl_dst.exists():
            print(f"File already exists, skip!!")
            continue

        lnk_img_path = find_image_for_label(images_src, stem)

        if lnk_img_path is None:
            print(" [ERROR!] Something went wrong: no image for the label! STOP! ")
            continue

                
        new_name_lnk_img = f"{stem}_{folder.name}{lnk_img_path.suffix}"
        img_dst = images_out / new_name_lnk_img

        shutil.copy2(label, lbl_dst)
        shutil.copy2(lnk_img_path, img_dst)

def labelled_folder_creation(base_dir, base_bag_name):
    images_out, labels_out = output_dirs_creation(base_dir)

    source_dirs = sorted(base_dir.glob(f"{base_bag_name}*"))

    for folder in source_dirs:
        images_src = folder / "images"
        labels_src = folder / "labels"

        if not labels_src.exists():
            print(f"[WARNING] No labels/ found in {folder.name}, skipping.")
            continue
            
        check_link_and_create_files(folder, images_src, labels_src, images_out, labels_out)
    