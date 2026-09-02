import pandas as pd
from pathlib import Path

LEDGER_PATH = Path(__file__).resolve().parent / "ledger.jsonl"

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', None)

COLUMNS = [
    'all/mAP50',
    'all/mAP50_95',
    'positives/recall',
    'negatives/far',
    'negatives/fp_per_neg_image',
]

def load_ledger(ledger_path = LEDGER_PATH):
    if not Path(ledger_path).is_file():
        raise FileNotFoundError(f"Ledger non trovato: {ledger_path}. Lancia prima benchmark.py")

    return pd.read_json(ledger_path, lines = True)

def per_run(df, benchmark, columns = COLUMNS):
    selected = df[df['benchmark'] == benchmark]
    present = [c for c in columns if c in selected.columns]

    return selected.set_index('run')[present].round(4)

def per_arm(df, benchmark, column):
    selected = df[df['benchmark'] == benchmark]

    return selected.pivot_table(index = 'arm', columns = 'tag', values = column).round(4)

# quanto ogni braccio perde rispetto al modello che non puo' aver dimenticato
# nulla: sul benchmark di forgetting e' la misura dell'oblio
def forgetting(df, benchmark, column = 'all/mAP50', reference = 'base'):
    table = per_arm(df, benchmark, column)

    if reference not in table.index:
        raise KeyError(f"Braccio di riferimento '{reference}' assente: {list(table.index)}")

    return (table.loc[reference] - table).round(4)

def main():
    df = load_ledger()

    for benchmark in sorted(df['benchmark'].unique()):
        print(f"\n=== {benchmark}")
        print(per_run(df, benchmark))

        if 'arm' not in df.columns:
            continue

        print("\nmAP50 per braccio x composizione")
        print(per_arm(df, benchmark, 'all/mAP50'))

        print("\nFAR per braccio x composizione")
        print(per_arm(df, benchmark, 'negatives/far'))

if __name__ == "__main__":
    main()
