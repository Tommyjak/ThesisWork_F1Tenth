from pathlib import Path

from core.loaders.base import Loader
from core.samples import ImageSample, LblStatus

class SubfoldersLoader(Loader):
    def __init__(self, root: Path):
        self.root = root

    def get_subfolders_names(self):
        subdir_names = []

        for dir in self.root.iterdir():
            if dir.is_dir():
                subdir_names.append(dir)

        return sorted(subdir_names)

    def find_samples(self):
        subdirs = self.get_subfolders_names()

        pos_samples = []
        neg_samples = []

        for subdir in subdirs:
            img_path = subdir / "images"
            lbl_path = subdir / "labels"

            for img in sorted(img_path.iterdir()):
                if not img.is_file():
                    continue

                file_name = img.stem

                lbl = lbl_path / f"{file_name}.txt"

                if not lbl.is_file():
                    neg_samples.append(ImageSample(
                        status = LblStatus.NEGATIVE,
                        ds_name = self.root.name,
                        img_path = img,
                        group = subdir.name,
                    ))
                elif lbl.stat().st_size == 0:
                    neg_samples.append(ImageSample(
                        status = LblStatus.NEGATIVE,
                        ds_name = self.root.name,
                        img_path = img,
                        lbl_path = lbl,
                        group = subdir.name,
                    ))
                else:
                    pos_samples.append(ImageSample(
                        status = LblStatus.POSITIVE,
                        ds_name = self.root.name,
                        img_path = img,
                        lbl_path = lbl,
                        group = subdir.name,
                    ))

        return pos_samples, neg_samples
