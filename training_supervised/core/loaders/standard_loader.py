from pathlib import Path

from core.loaders.base import Loader
from core.samples import ImageSample, LblStatus

class StandardLoader(Loader):
    def __init__(self, root: Path):
        self.root = root

    def find_samples(self):
        img_path = self.root / "images"
        lbl_path = self.root / "labels"

        pos_samples = []
        neg_samples = []

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
                ))

            elif lbl.stat().st_size == 0:
                neg_samples.append(ImageSample(
                    status = LblStatus.NEGATIVE,
                    ds_name = self.root.name,
                    img_path = img,
                    lbl_path = lbl,
                ))
            else:
                pos_samples.append(ImageSample(
                    status = LblStatus.POSITIVE,
                    ds_name = self.root.name,
                    img_path = img,
                    lbl_path = lbl,
                ))

        return pos_samples, neg_samples
