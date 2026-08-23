from dataclasses import dataclass
from pathlib import Path

from preprocessing.core.samples import ImageSample, LblStatus, Split

@dataclass
class Dataset:
    name: str
    root: Path
    pos_samples: list[ImageSample]
    neg_samples: list[ImageSample]

    def get_positive_samples(self):
        return self.pos_samples

    def get_negative_samples(self):
        return self.neg_samples

    def add_sample(self, sample: ImageSample):
        if sample.is_annotated():
             self.pos_samples.append(sample)
        else:
             self.neg_samples.append(sample)

    def get_pos_count(self):
        return len(self.pos_samples)

    def get_neg_count(self):
        return len(self.neg_samples)

    def get_split(self, split: Split, status: LblStatus):
        returnable = [] 
        samples = []
        if status == LblStatus.POSITIVE:
            samples = self.pos_samples
        else:
            samples = self.neg_samples

        for sample in samples:
            if sample.split == split:
                returnable.append(sample)

        return returnable

    def count(self, split: Split, status: LblStatus):
        bucket = self.get_split(split, status)
        return len(bucket)