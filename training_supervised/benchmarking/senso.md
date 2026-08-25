```
Flusso

  benchmarks.yaml
        │
        ▼
   loading.py ──► Dataset (pos + neg, interi, nessuno split)
        │
        ▼
  prediction.py ──► inferenza sul modello          [GPU, lenta, UNA volta]
        │
        ▼
  evaluation/predictions/<run>/<bm>.jsonl   ← predizioni + ground truth grezze
        │
        ├──► slices.py    partiziona i record
        ▼
    metrics.py ──► mAP / P / R / FAR per fetta      [CPU, secondi, N volte]
        │
        ▼
  evaluation/ledger.jsonl  ← una riga per (run × benchmark), colonne per fetta

  Il punto della struttura è la linea orizzontale in mezzo. Sopra c'è tutto ciò che costa GPU e va fatto una volta
  sola per modello; sotto c'è tutto ciò che costa secondi e puoi rifare quante volte vuoi. Aggiungere la fetta "solo
  oggetti piccoli" o "solo frame in controluce" tocca solo la metà inferiore: nessuna re-inferenza, nessun
  riallenamento, e vale retroattivamente su tutti i checkpoint già valutati.
```

Responsabilità, file per file


- configs/benchmarks.yaml — stesso schema di datasets.yaml (name, root, type), chiave benchmarks:. Sono dataset di sola valutazione: non entrano mai in training, quindi non hanno pos_ratio né val_pct.


- loading.py — importa preprocessing.loaders e preprocessing.core, restituisce una lista di Dataset interi. È l'unico punto di contatto con preprocessing/, ed è deliberato: i loader e le strutture dati sono convenzioni di filesystem condivise, duplicarle significherebbe che una correzione applicata da una parte resta rotta dall'altra. Config, metriche e logica di partizione invece sono duplicate, perché sono politiche che devono poter divergere.

- prediction.py — l'unico che tocca la GPU. Per ogni immagine del benchmark scrive un record: 
``` yaml
{"image": "/…/frame_0421.jpg", "w": 1280, "h": 720,

 "pred": [{"box": [x1,y1,x2,y2], "score": 0.87}],

 "gt":   [{"box": [x1,y1,x2,y2]}]}
```
Gira a conf bassissima (0.001) e senza soglia operativa: filtrare è compito delle metriche, non dell'inferenza. Salvando anche la ground truth accanto alle predizioni, il calcolo delle metriche diventa autosufficiente — non deve rileggere i file .txt né conoscere le convenzioni dei loader.

- slices.py — definisce le partizioni. Due livelli che non vanno confusi:


    - per immagine: dominio, sessione di registrazione, positivi vs negativi, numero di avversari in scena

    - per oggetto: fascia di area del box, cioè la distanza dell'avversario


    Il secondo è quello che vale di più nel tuo caso, e non si esprime come filtro sui record: richiede di filtrare i box dentro ogni record, sia nella ground truth che nelle predizioni.


- metrics.py — da record filtrati a numeri. Tre gruppi: mAP50-95 e mAP50, precision/recall/TP/FP/FN alla soglia operativa, false alarm rate sui negativi.


- ledger.py — append di una riga JSONL per (run, benchmark), con colonne nominate <slice>/<metrica>. JSONL e non CSV perché il numero di fette cambia nel tempo e un DictWriter si romperebbe alla prima riga con una colonna in più.


- pipeline.py — run_benchmark(weights, run_name): carica i benchmark, chiama prediction.py se il dump non esiste già, poi calcola le metriche su tutte le fette e scrive nel ledger.


- benchmark.py — scorre results/*/weights/best.pt e chiama run_benchmark per ciascuno. Va lanciato anche sul modello di partenza, che è la riga zero del ledger contro cui misuri il forgetting.

``` bash
python3 training_supervised/train.py        # allena

python3 training_supervised/benchmark.py    # valuta
```