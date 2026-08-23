from dataclasses import dataclass
from pathlib import Path
from enum import Enum

class LblStatus(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"

class Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

@dataclass
class ImageSample:
    status: LblStatus
    ds_name: str
    img_path: Path
    lbl_path: Path | None = None
    split: Split | None = None

    def is_annotated(self):
        return self.status == LblStatus.POSITIVE

    def sample_name(self):
        return self.img_path.stem