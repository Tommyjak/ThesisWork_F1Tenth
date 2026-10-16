#!/bin/bash
# Punta i percorsi che analysis.py e plots.py si aspettano alla campagna scelta.
#
#   ./use_campaign.sh main         la campagna principale (default)
#   ./use_campaign.sh sequential   il continual learning a 4 passi
#   ./use_campaign.sh              mostra quale e' attiva
#
# I dati veri stanno in campaigns/<nome>/; qui si spostano solo quattro
# collegamenti simbolici, quindi cambiare campagna e' istantaneo e non copia
# nulla. Il codice resta ignaro: continua a leggere results/, references/,
# evaluation/ledger.jsonl ed evaluation/predictions come sempre.

set -euo pipefail
BASE="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE"

if [ $# -eq 0 ]; then
    echo "campagna attiva: $(basename "$(dirname "$(readlink -f results)")")"
    echo
    echo "disponibili:"
    for c in campaigns/*/; do
        n=$(basename "$c")
        printf "  %-26s %5s righe di ledger, %3s run\n" "$n" \
            "$(wc -l < "$c/ledger.jsonl" 2>/dev/null || echo '-')" \
            "$(ls "$c/results" 2>/dev/null | wc -l)"
    done
    exit 0
fi

C="$1"
[ -d "campaigns/$C" ] || { echo "campagna '$C' inesistente. Disponibili: $(ls campaigns | tr '\n' ' ')"; exit 1; }

ln -sfn "campaigns/$C/results"    results
ln -sfn "campaigns/$C/references" references
ln -sfn "../campaigns/$C/ledger.jsonl" evaluation/ledger.jsonl
ln -sfn "../campaigns/$C/predictions"  evaluation/predictions

echo "campagna attiva: $C  ($(wc -l < "campaigns/$C/ledger.jsonl") righe, $(ls "campaigns/$C/results" | wc -l) run)"
