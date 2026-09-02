import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from preprocessing.loading import load_datasets
import preprocessing.configs.exp_yaml_parser as eyp

def get_totals():
    totals = {}

    for dataset in load_datasets():
        totals[dataset.name] = (dataset.get_pos_count(), dataset.get_neg_count())

    return totals

def report_ratios():
    totals = get_totals()

    for name in totals:
        print(f"{name:32s} pos={totals[name][0]:5d} neg={totals[name][1]:5d}")

    print()

    for exp in eyp.get_exp_infos():
        pos = 0
        neg = 0

        for name, (pos_ratio, neg_ratio) in exp['ratios'].items():
            pos += round(totals[name][0] * pos_ratio)
            neg += round(totals[name][1] * neg_ratio)

        share = 100 * pos / (pos + neg)

        print(f"{exp['name']:18s} pos={pos:5d} neg={neg:5d}  ->  {share:.1f}:{100 - share:.1f}")

if __name__ == "__main__":
    report_ratios()
