import os
import sys
import shutil
from pathlib import Path

BASE_DIR = Path("/home/tm/Documents/universita/tirocinio_hipert/research/datasets/squeaky_17-04-26")

def labelled_folder_creation(base_dir):
    images_out = BASE_DIR / "complete_labels" / "images"
    labels_out = BASE_DIR / "complete_labels" / "labels"

    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    source_dirs = sorted(BASE_DIR.glob("2Macchine_*"))

    for folder in source_dirs:
        folder_name = folder.name
        images_src = folder / "images"
        labels_src = folder / "labels"

        if labels_src.exists():
            for label in sorted(labels_src.iterdir()):
                if not label.is_file():
                    continue

                stem = label.stem
                suffix = label.suffix

                new_name = f"{stem}_{folder_name}{suffix}"
                print(new_name)

                lbl_dst = labels_out / new_name

                if lbl_dst.exists():
                    print(f"File already exists, skip!!")
                    continue

                lnk_img_path = images_src / f"{stem}.jpg"
                if not lnk_img_path.exists():
                    print(" [ERROR!] Something went wrong: no image for the label! STOP! ")
                    break

                
                new_name_lnk_img = f"{stem}_{folder_name}.jpg"
                img_dst = images_out / new_name_lnk_img

                shutil.copy2(label, lbl_dst)
                shutil.copy2(lnk_img_path, img_dst)

def main(base_dir):
    labelled_folder_creation(base_dir)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
        main(base_dir)
    else:
        print("No base directory provided. Using default path.")
        main(BASE_DIR)
