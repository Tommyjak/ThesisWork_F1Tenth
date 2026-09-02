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

il benchmark di forgetting deve venire dalla vecchia distribuzione, cioè dall'ambiente su cui i modelli base sono stati allenati — stesso laboratorio, stessa pista, stessa macchina — ma da sessioni mai viste. Se raccogli un terzo ambiente nuovo, quello che misuri è generalizzazione, non forgetting. Sono due cose diverse e servono due benchmark distinti:

```
┌───────────────────────────────┬────────────────────────────────────────┬──────────────────┐
│           benchmark           │             distribuzione              │   cosa misura    │
├───────────────────────────────┼────────────────────────────────────────┼──────────────────┤
│ icra_holdout                  │ nuovo ambiente, già usato in fine-tune │ adattamento      │
├───────────────────────────────┼────────────────────────────────────────┼──────────────────┤
│ il nuovo che raccogli         │ vecchio ambiente, mai visto            │ forgetting       │
├───────────────────────────────┼────────────────────────────────────────┼──────────────────┤
│ (opzionale) un terzo ambiente │ mai visto, mai allenato                │ generalizzazione │
└───────────────────────────────┴────────────────────────────────────────┴──────────────────┘
```

Raccogliendolo ex-novo eviti del tutto la contaminazione e non devi riallenare niente. È la soluzione migliore di quella che ti avevo proposto.

---

Sulla seconda domanda: "Pseudo" è un aggettivo che ti stai dando contro. Quello che descrivi ha un nome preciso in letteratura ed è domain-incremental continual learning, in modalità offline.
Le tre parole servono tutte:
- domain-incremental perché l'insieme delle classi non cambia mai (hai sempre e solo racecar), cambia la distribuzione dell'input — luce, pista, telecamera, sfondo. È il contrario di class-incremental, dove aggiungi classi nuove. La tua
  è la variante giusta per la guida autonoma, ed è anche quella su cui il forgetting si manifesta in modo più subdolo: il modello non "dimentica una classe", degrada silenziosamente sui vecchi contesti.
- continual perché il protocollo è sequenziale e ad accesso limitato: parti da un checkpoint, arriva un nuovo dominio, ti riallena senza (o con poco) accesso ai dati vecchi. È esattamente la definizione.
- offline perché i dati arrivano in blocchi, li etichetti, e alleni in batch. Questa non è una debolezza: la stragrande maggioranza della letteratura di CL è offline, con checkpoint e esperimenti riproducibili. L'online/streaming learning è un'altra sotto-area.

Quindi no, non è improprio chiamarlo continual learning. Quello che è improprio è chiamarlo on-device o lifelong: il training avviene su una macchina con GPU, non sul Jetson, e l'etichettatura ha un umano in mezzo. Vanno dette entrambe,
ma non tolgono niente.


L'unica cosa che oggi indebolisce davvero la definizione è un'altra, e non è quella che pensavi:
la tua sequenza ha lunghezza 2. Base → +icra. Con due passi il forgetting lo osservi ma non lo vedi accumulare, e l'accumulo è il fenomeno interessante — è quello che distingue "il fine-tuning costa qualcosa" da "il fine-tuning ripetuto degrada il modello fino a renderlo inutile". Se con la pipeline di docker_humble/ raccogli un secondo ambiente e fai base → +A → +B, misurando dopo ogni passo su tutti i benchmark precedenti, ottieni la matrice R_{i,j} standard del continual learning (performance sul task j dopo aver appreso il task i) e da lì i due numeri canonici: backward transfer (quanto perdi sul vecchio) e forward transfer (quanto il vecchio aiuta sul nuovo). Quello è continual learning senza aggettivi difensivi.


E vale la pena dirlo esplicitamente: la pipeline rosbag → mp4 → frame → dataset è una componente a sé, non la parte di learning. In tesi va presentata per quello che è — l'infrastruttura di acquisizione che rende la sequenza possibile e ripetibile — ed è un contributo autonomo, perché è il pezzo che manca nella maggior parte dei lavori di CL, dove i "task" sono partizioni artificiali di CIFAR. Tu i domini li acquisisci davvero dal veicolo. Chiamarla "pseudo" nasconde proprio la parte più difendibile del lavoro.

---

Il principio: tre tracce che non si bloccano a vicenda


La cosa importante da vedere è che i tre pezzi di lavoro rimasti sono indipendenti. Nessuno aspetta gli altri, e solo l'analisi finale li ricongiunge. Se li fai in sequenza perdi due settimane per niente.  
```
┌──────────────────────────────────────────┬───────────────────────────┬────────────┬───────────────────┐
│                 traccia                  │           dove            │ dipende da │      blocca       │
├──────────────────────────────────────────┼───────────────────────────┼────────────┼───────────────────┤
│ A — codice di training                   │ locale + freeman          │ niente     │ i 6 nuovi modelli │
├──────────────────────────────────────────┼───────────────────────────┼────────────┼───────────────────┤
│ B — codice di benchmarking               │ solo locale               │ niente     │ l'analisi         │
├──────────────────────────────────────────┼───────────────────────────┼────────────┼───────────────────┤
│ C — raccolta del benchmark di forgetting │ macchina + docker_humble/ │ niente     │ l'analisi         │
└──────────────────────────────────────────┴───────────────────────────┴────────────┴───────────────────┘
```

La traccia B non ha bisogno della GPU: icra_holdout sono 886 immagini, yolo26n su CPU le processa in un paio di minuti. Puoi scrivere e validare tutta la catena di benchmarking sul portatile mentre freeman allena.

---


Traccia A — sbloccare il training (mezza giornata, poi è GPU che gira da sola)
A1. Decommentare train.py:109. È ancora lì.
A2. Tre chiavi nuove in experiments.yaml: init_from, hyp, e i metadati arm/tag. Più i corrispondenti getter in exp_yaml_parser.py, e in train.py il passaggio di init_weights a train_model più l'ordinamento che mette in fondo gli
esperimenti con init_from.
A3. Un unico experiments.yaml con tutti e dodici gli esperimenti. Non rinominare neg_full / neg_balanced / neg_few: la guardia if (RESULTS_DIR / run_name).exists() li salta automaticamente, e ti risparmi 6 ore di GPU inutili. Aggiungi
solo arm: joint e tag: full come metadati. Questa proprietà — aggiungi esperimenti, rilanci train.py, allena solo quelli nuovi — è già nel codice ed è quella che ti fa lavorare in modo incrementale per le prossime settimane.
A4. Lancio in tmux su freeman. Prima passata: i tre base_* (~6h). Seconda passata: i sei ft_* e ftr_* (~2h). Due lanci perché i fine-tune hanno bisogno dei best.pt dei base.
I ratio
Togliendo icra_base i totali cambiano (pos 3593→3193, neg 6305→4039), quindi i moltiplicatori vanno ricalcolati per tenere gli stessi rapporti obiettivo:
base_* — tutto tranne icra_base:

```
┌───────────────┬────────┬─────────┬─────────┬─────────┐
│               │ huawei │ squeaky │ objects │ pos:neg │
├───────────────┼────────┼─────────┼─────────┼─────────┤
│ base_full     │ 1.0    │ 1.0     │ 1.0     │ 44:56   │
├───────────────┼────────┼─────────┼─────────┼─────────┤
│ base_balanced │ 0.868  │ 0.868   │ 0.6     │ 50:50   │
├───────────────┼────────┼─────────┼─────────┼─────────┤
│ base_few      │ 0.355  │ 0.355   │ 0.6     │ 65:35   │
└───────────────┴────────┴─────────┴─────────┴─────────┘                           
```

(f1tenth resta 1.0 ovunque: ha zero negativi.)
ft_* — solo icra_base (404 pos / 2266 neg):

```
┌─────────────┬───────────┬─────────┐
│             │ neg_ratio │ pos:neg │
├─────────────┼───────────┼─────────┤
│ ft_full     │ 1.0       │ 15:85   │
├─────────────┼───────────┼─────────┤
│ ft_balanced │ 0.178     │ 50:50   │
├─────────────┼───────────┼─────────┤
│ ft_few      │ 0.096     │ 65:35   │
└─────────────┴───────────┴─────────┘
```

ftr_* (replay) — la composizione di ft_X più tutti gli altri dataset con {pos_ratio: 0.1, neg_ratio: 0.1 × ratio di base_X}. Circa 630 immagini vecchie contro 2670 nuove, cioè ~19% di replay.
Iperparametri del fine-tune, uguali per ft_* e ftr_*:
hyp: {epochs: 30, lr0: 0.0001, warmup_epochs: 0, patience: 10}
Con lr0: 0.001 e 100 epoch su 404 positivi non stai fine-tunando, stai riallenando da un'inizializzazione fortunata — e misureresti un fenomeno diverso.

---

Traccia B — scrivere il benchmarking (in parallelo, sul portatile)
L'ordine è quello di prima, con una modifica: valida ogni pezzo sui tre modelli che hai già. Non aspettare i sei nuovi.
1. group su ImageSample e nei due loader.
2. prediction.py → gira su neg_full + icra_holdout, produci un JSONL, aprilo e guarda tre record a mano.
3. metrics.py con la sola fetta all → controllo di sanità obbligatorio: la mAP50 che ottieni su icra_holdout dev'essere plausibilmente più bassa di 0.975 (il val set di training). Se ti viene 0.99 o 0.3, il matching è sbagliato. Con una
   fetta sola te ne accorgi; con dodici, no.
4. ledger.py, pipeline.py, benchmark.py.
5. slices.py per ultimo.
Alla fine di questa traccia hai un ledger con 4 righe (yolo26n + i tre neg_*) su icra_holdout. Quelle quattro righe sono già un risultato di tesi: dicono quale composizione di negativi generalizza meglio a nuove sessioni dello stesso
ambiente. È la domanda che ti eri posto all'inizio, e la chiudi prima ancora di iniziare il continual learning.

---

Traccia C — il benchmark di forgetting
Il vincolo da non sbagliare: stesso ambiente dei dataset base, sessioni mai viste. Concretamente, registrazioni nuove nello stesso laboratorio/pista di squeaky_17-04-26, non un posto nuovo.
Dimensione minima ragionevole: 2-3 sessioni, ~300 positivi e ~500 negativi. Sotto i 200 positivi gli intervalli di confidenza sulla mAP diventano più larghi delle differenze che vuoi misurare, e non concludi niente.
Struttura sul disco, identica a quella che hai già:

```
datasets/forgetting_holdout/
├── sessione_1/{images,labels}
├── sessione_2/{images,labels}
└── sessione_3/{images,labels}
```

Poi una voce in benchmarks.yaml con type: subfolders ed enabled: true. Zero codice.
Se puoi, includi qualche sessione con le scatole/distrattori: dopo il fine-tune su solo icra è lì che il modello ricomincerà ad allucinare, e la FAR su quelle sessioni è il numero più parlante che avrai.

---

Quando le tracce si ricongiungono
Un solo comando, e il ledger si popola per intero:
python3 training_supervised/benchmark.py
12 modelli × 2 benchmark = 24 righe. prediction.py salta i dump già fatti, quindi rilanciarlo dopo aver aggiunto una fetta a slices.py costa secondi e ricalcola tutto retroattivamente.
Da lì l'analisi è un pandas.read_json(lines=True) e un raggruppamento su arm × tag:
- arm risponde alla domanda principale — retraining congiunto contro fine-tuning sequenziale contro replay;
- tag ti dice se la risposta cambia al variare della quantità di negativi, che è l'asse che avevi già esplorato.

---

Due cose da non fare
Non aspettare la traccia C per iniziare la A. I sei modelli nuovi non dipendono dal benchmark di forgetting: si allenano tutti su dati che hai già. Quando i dati nuovi arrivano, devono trovare i modelli pronti e il codice di valutazione
scritto, altrimenti hai un collo di bottiglia da una settimana alla fine.
Non allargare gli assi. Hai due variabili (composizione dei negativi × strategia di training) per 12 run, e sono già tanti. La tentazione di aggiungere imgsz, o yolo26s, o un terzo valore di replay, va rimandata a dopo che hai una
tabella completa. Un esperimento incompleto su tre assi vale meno di uno completo su due.

--- 

` Come funziona, seguendo il flusso
`

  benchmark.py — il punto di contatto con l'esterno

  Il file di ingresso non conosce né benchmark né metriche: sa solo dove stanno i pesi e dove va il ledger. Questo è deliberato — è l'unico file che tocchi quando cambia il layout delle cartelle, e non contiene una riga di logica di
  valutazione.

  collect_models() costruisce una lista di coppie (run_name, weights) da due sorgenti: i baseline in models/ (oggi solo yolo26n.pt) e ogni results/<run>/weights/best.pt. I baseline sono la riga zero del ledger — il modello non allenato
  contro cui misuri quanto vale tutto il resto. Se un best.pt manca stampa e salta invece di sollevare: con dodici run e un training ancora in corso su freeman, vuoi valutare quelli pronti senza aspettare gli altri.

  main() chiama run_benchmark una volta per modello. Il ciclo è al livello più esterno perché ogni iterazione è indipendente: puoi interromperlo con Ctrl-C dopo il terzo modello e il ledger contiene tre righe corrette.

  run_benchmark — dove le due metà si incontrano

  È la funzione che materializza la linea orizzontale di senso.md. Sopra c'è la GPU, sotto la CPU, e questa funzione le attraversa una volta per benchmark:
```
  load_benchmarks()  →  predict_benchmark()  →  evaluate_dump()  →  build_row()  →  append_row()
       yaml               GPU, minuti            CPU, secondi        dict piatto      disco
```

  load_benchmarks() legge benchmarks.yaml e restituisce oggetti Dataset interi — nessuno split, perché un benchmark non si divide in train/val: è tutto test.

  predict_benchmark() è la parte che già conosci: scrive evaluation/predictions/<run>/<bm>.jsonl e salta se il dump esiste già ed è coerente con lo sha dei pesi. Da qui in avanti nessuno tocca più immagini, GPU, o file .txt di label:
  tutto ciò che serve alle metriche è dentro il JSONL. È la proprietà che rende ripetibile la metà inferiore.

  extra è il gancio per arm e tag. Oggi è None e le colonne non compaiono; quando aggiungerai quelle chiavi a experiments.yaml, benchmark.py le legge e le passa qui, e finiscono nella riga senza toccare nient'altro. È il punto in cui la
  traccia B si aggancia alla A.

  evaluate_dump — dal file ai numeri

  Tre passi, e l'ordine conta.

  read_dump() porta in memoria una lista di dizionari. Su 886 immagini è circa 1 MB: non serve streaming, e avere tutto in RAM è ciò che permette di calcolare dodici fette senza rileggere il file dodici volte.

  build_slices(records) costruisce le fette guardando i dati. Le prime sei sono fisse (all, positives, negatives, area_small/medium/large), le ultime dipendono dai group effettivamente presenti nel dump. È il motivo per cui group doveva
  stare su ImageSample: senza, l'ultimo ciclo non produce niente e perdi la fetta per sessione — che è quella su cui poi farai il bootstrap a blocchi.

  Poi un ciclo che per ogni fetta chiama apply_slice e compute_metrics, e mette il risultato in un dizionario nome_fetta → metriche.

  apply_slice — perché una fetta è un dizionario e non una funzione

  Una fetta qui è un dato: {'name': ..., 'status': ..., 'group': ..., 'band': ...}. Avrei potuto rappresentarla come funzione, ma allora le fette per gruppo — che nascono a runtime — avrebbero richiesto closure o partial, e leggere il
  file diventava più difficile. Come dizionario, aggiungere una fetta è aggiungere una riga a build_slices, e apply_slice resta tre if in fila.

  I tre filtri si compongono: i campi a None non filtrano niente. new_slice('positives', status='positive') tiene solo i positivi; new_slice('area_large', band=...) tiene tutte le immagini ma pota i box.

  E qui c'è la differenza che vale la pena capire. filter_by_status e filter_by_group scartano immagini; filter_by_band scarta box dentro le immagini. Il secondo caso è quello che ti interessa davvero: "quanto è bravo sugli avversari
  lontani" non è una domanda su un sottoinsieme di frame, è una domanda su un sottoinsieme di oggetti, e gli oggetti piccoli e grandi convivono nello stesso frame.

  Due dettagli non ovvi in filter_by_band:

  - trimmed = dict(record) fa una copia superficiale, poi rimpiazza gt e pred con liste nuove. Senza la copia, potare i box della fetta area_small distruggerebbe i record originali e le fette successive lavorerebbero su dati mutilati. È
    il bug silenzioso che questa riga previene.
  - le immagini restano tutte, anche quelle che finiscono con zero box. Così n_images è confrontabile fra le fette, e i negativi restano dentro — quindi la loro FAR viene calcolata per fascia di area, e ti dice se le allucinazioni del
    modello sono grandi o piccole. Sui dati veri: area_small FAR 0.00, area_medium 0.047, area_large 0.62. Il modello vecchio non inventa macchinine lontane, inventa macchine enormi in primo piano. È esattamente il tipo di informazione
    che una metrica aggregata nasconde.

  Le bande sono quelle COCO (32² e 96² pixel), su area in pixel dell'immagine originale — citabili in tesi senza doverle giustificare.

  Un limite che dichiaro apertamente: COCO, nella valutazione per area, ignora le predizioni che matchano ground truth fuori banda invece di contarle come FP. Qui filtro ambo i lati per area propria, che è più semplice e più conservativo
  — leggi le fette per area come confronti relativi fra modelli, non come mAP COCO ufficiale.

  compute_metrics — tre gruppi di numeri, e i None

  Restituisce un dizionario piatto con conteggi (n_images, n_pos, n_neg, n_gt), mAP, punto operativo e false alarm rate. Il principio che governa tutto: una metrica non definita vale None, mai 0.

  Sulla fetta negatives non ci sono ground truth, quindi mAP e recall non esistono. Scriverci 0 significherebbe leggere nel ledger "il modello ha mAP 0 sui negativi" e concludere che è pessimo, quando la verità è che la domanda non ha
  senso. None diventa null nel JSON e NaN in pandas, che è ciò che vuoi in fase di analisi. Puoi verificarlo nella tabella qui sotto: la riga negatives ha - su mAP e recall ma FAR piena, la riga positives ha il contrario.

  compute_map

  count_gt prima di tutto: se il denominatore è zero, esce subito con (None, None).

  Poi un ciclo sulle dieci soglie IoU da 0.50 a 0.95. Per ognuna:

  collect_entries fa il matching immagine per immagine e poi appiattisce tutto. Il punto sottile è che il matching è locale ma la curva PR è globale: una detection viene confrontata solo con i gt della sua immagine — ovvio — ma poi il suo
  (score, tp/fp) finisce in una lista unica riordinata per confidenza su tutto il benchmark. È così che si costruisce una curva precision-recall: abbassando progressivamente la soglia, ogni detection in ordine di confidenza sposta il
  punto sulla curva.

  match_preds è greedy: predizioni dalla più sicura alla meno, ciascuna prende il gt libero con IoU massima. L'array matched garantisce che un gt sia consumato una volta sola — la seconda detection sulla stessa macchina è un falso
  positivo, che è la definizione corretta e il motivo per cui l'NMS conta. Non è l'assegnamento ottimo globale (Hungarian), ma è ciò che fanno COCO, VOC e ultralytics, quindi i numeri sono comparabili con la letteratura.

  compute_ap accumula TP e FP nell'ordine, costruisce recall e precision punto per punto, applica l'inviluppo (scorrendo da destra rende la precisione monotona decrescente: elimina i denti di sega dovuti a singole detection fortunate) e
  integra col metodo dei trapezi. Parte da (recall=0, precision=1) perché la curva deve toccare l'asse.

  mAP50 è l'AP alla prima soglia, mAP50_95 la media delle dieci. Con una classe sola "mAP" e "AP" coincidono; tengo il nome mAP per allinearmi a come lo chiama ultralytics nei log.

  compute_operating

  Stesso matching, ma a IoU fissa 0.5 e solo sulle predizioni sopra CONF_OP = 0.25. Questa è la domanda diversa: non "quanto è buono il ranking del modello" ma "quante macchine vede e quante ne inventa se lo metti sulla macchina adesso".
  fn = len(gt) - hits per immagine, quindi un frame senza detection contribuisce tutti i suoi gt ai falsi negativi.

  conf_op finisce come colonna nel ledger: fra due mesi la prima domanda davanti a un numero strano sarà "a che soglia?", e la risposta è nella riga stessa.

  compute_far

  Seleziona i negativi della fetta e restituisce due numeri complementari:

  - far — frazione di immagini negative con almeno una detection. È la metrica di sicurezza: quante volte il pianificatore frenerebbe per un fantasma.
  - fp_per_neg_image — media delle detection per immagine negativa. Distingue "sbaglia raramente ma male" da "sbaglia spesso con un box solo".

  Se la fetta non ha negativi restituisce None su entrambi, non 0.

  build_row e append_row — dal dizionario annidato alla tabella

  flatten_metrics appiattisce slice_results in chiavi <fetta>/<metrica>: all/mAP50, negatives/far, area_large/precision. È il compromesso fra "una tabella che pandas legge in una riga" e "un numero di colonne che cambia nel tempo" — ed è
  precisamente il motivo per cui il ledger è JSONL e non CSV: un DictWriter si romperebbe la prima volta che aggiungi una fetta, mentre qui pandas.read_json(lines=True) mette semplicemente NaN nelle colonne che le righe vecchie non
  avevano.

  I metadati vengono dal .meta.json scritto da prediction.py, non ricalcolati: weights_sha nel ledger è lo stesso sha che ha certificato il dump, quindi una riga di metriche è tracciabile fino ai byte esatti del checkpoint che l'ha
  prodotta.

  append_row non è un append puro: legge il ledger, elimina la riga con la stessa coppia (run, benchmark), poi aggiunge in fondo. Vuol dire che rilanciare benchmark.py aggiorna invece di duplicare. È la proprietà che rende usabile il
  ciclo di lavoro descritto in senso.md: aggiungi una fetta a slices.py, rilanci, l'inferenza viene saltata perché i dump sono validi, e in pochi secondi tutte le righe si ricalcolano retroattivamente su tutti i checkpoint già valutati.
  Senza la deduplica ti ritroveresti con dodici copie di ogni run e nessun modo di sapere quale è aggiornata.

  ---

  Validazione

  Confronto con l'implementazione di riferimento. compute_ap contro ultralytics.utils.metrics.compute_ap su sei casi casuali: differenza massima 8·10⁻⁴, che è la distanza fra interpolazione continua e a 101 punti. Poi l'intera catena
  contro model.val() di ultralytics sullo stesso identico dataset:

```
  ┌─────────────────────────┬────────┬──────────┐
  │                         │ mAP50  │ mAP50-95 │
  ├─────────────────────────┼────────┼──────────┤
  │ ultralytics model.val() │ 0.0912 │ 0.0238   │
  ├─────────────────────────┼────────┼──────────┤
  │ questo codice           │ 0.0896 │ 0.0236   │
  └─────────────────────────┴────────┴──────────┘
```

  Delta 0.0016 e 0.0002. Il codice calcola quello che dice di calcolare.

  Il ledger prodotto (11-05-26_best su icra_holdout, 886 immagini, 46 s su CPU):

```
  slice              n_images  n_pos  n_neg  n_gt   mAP50  mAP50_95   prec  recall     tp    fp    fn     far
  all                     886    295    591   349  0.0896    0.0236  0.153   0.418    146   808   203   0.657
  positives               295    295      0   349  0.2136    0.0562  0.340   0.418    146   283   203       -
  negatives               591      0    591     0       -         -  0.000       -      0   525     0   0.657
  area_small              886    295    591     0       -         -      -       -      0     0     0   0.000
  area_medium             886    295    591    22  0.0481    0.0131  0.061   0.455     10   153    12   0.047
  area_large              886    295    591   327  0.0532    0.0135  0.118   0.284     93   698   234   0.619
  group_test_camera_2     329    103    226   150  0.0792    0.0205  0.143   0.340     51   306    99   0.491
  group_test_camera_4     557    192    365   199  0.0984    0.0266  0.159   0.477     95   502   104   0.759
```

  Sul controllo di sanità di senso.md. Il valore 0.0896 è ben sotto lo 0.975 del val set, ma è anche sotto il "0.3 = matching rotto" che avevo scritto. Ho verificato che non è un bug: la conversione GT ricalcolata a mano su frame_001224
  combacia al centesimo di pixel, l'area mediana dei box predetti (0.018 dell'immagine) è coerente con quella dei gt (0.021), e ultralytics indipendentemente dà lo stesso numero. È un risultato vero: 11-05-26_best non ha mai visto
  l'ambiente ICRA, allucina su due negativi su tre, e 525 falsi positivi su 591 immagini vuote distruggono la precisione — quindi l'area sotto la curva PR crolla anche con una recall del 42%.

  Detto altrimenti: questa riga è la baseline di domain gap della tua tesi. È il numero contro cui misurerai quanto vale addestrare su ICRA, e più è basso più il tuo contributo è visibile. La soglia di allarme da tenere è quella alta
  (0.99 = ground truth che coincide con le predizioni per errore); un numero basso su un modello fuori dominio è il comportamento atteso.

  ---
