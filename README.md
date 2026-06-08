# 🎯 Il Dottore sotto esame — Machine Learning applicato ad *Affari Tuoi*

> Progetto per il corso di **Elementi di Intelligenza Artificiale**  
> Università degli Studi di Napoli Federico II — A.A. 2023/2024  
> Prof. Giancarlo Sperlì

**Studenti:** Crisci Francesco Pio (N46007601) · Raia Umberto (N46007558)

---

## 📌 Descrizione

*Affari Tuoi* è il programma televisivo più visto d'Italia. Al centro dello show c'è il **Dottore**, che propone offerte in denaro al concorrente in cambio del suo pacco. Ma le sue offerte sono eque? Seguono una logica? E soprattutto — quanto sta trattenendo rispetto a quello che la matematica suggerirebbe?

Questo progetto risponde con gli strumenti del **Machine Learning supervisionato**: a partire da 89 puntate reali (12 febbraio – 20 maggio 2024), vengono addestrati modelli in grado di classificare le offerte del Dottore e di stimarne il valore esatto in euro.

L'analisi rivela che le offerte del Dottore **non sono casuali** — seguono pattern statisticamente riconoscibili, appresi da un algoritmo e replicabili su partite mai viste. Il modello è così accurato da poter essere usato come **Dottore artificiale** in una simulazione interattiva da terminale.

---

## 🗂️ Struttura del repository

```
AffariTuoi/
│
├── data/
│   └── azioni_partite.json          # Dataset originale (89 puntate, fonte: luzo7/affari-tuoi)
│
├── python/
│   ├── affari_tuoi_ml.py            # Script ML principale (classificazione + regressione)
│   └── output/
│       ├── dataset_offerte.csv      # 386 offerte con feature calcolate
│       ├── risultati_modelli.csv    # Accuracy, precision, recall, F1 per ogni modello
│       ├── risultati_regressione.csv# Predizioni vs offerte reali in euro
│       ├── grafico_accuracy.png
│       ├── matrici_confusione.png
│       ├── feature_importance.png
│       ├── ratio_per_turno.png
│       ├── distribuzione_classi.png
│       ├── regressione_scatter.png
│       └── regressione_errori.png
│
├── game/
│   ├── gioca.py                     # Simulatore da terminale con il Dottore artificiale
│   ├── modello_dottore.pkl          # Modello serializzato (generato da affari_tuoi_ml.py)
│   └── feature_cols.pkl             # Ordine delle feature (generato da affari_tuoi_ml.py)
│
├── knime/
│   └── AffariTuoi/                  # Workflow KNIME 5 (Random Forest Learner + Scorer)
│
├── latex/
│   ├── tesina_affari_tuoi.tex       # Sorgente LaTeX
│   ├── tesina_affari_tuoi.pdf       # Tesina compilata
│   └── img/                         # Grafici e screenshot usati nella tesina
│
└── README.md
```

---

## 🧠 Approccio ML

### Il problema
Per ogni offerta del Dottore si calcola il **ratio**:

```
ratio = offerta_dottore / valore_atteso_premi_rimasti
```

Un ratio = 1 significa offerta perfettamente equa. Il dataset mostra una media di **0.789** — il Dottore trattiene in media il 21% del valore atteso. Il comportamento non è uniforme: è più aggressivo al **3° turno** (ratio 0.729) e più generoso nelle fasi finali (0.893).

### Classificazione — tre classi di offerta

| Classe | Soglia | Significato |
|--------|--------|-------------|
| **Bassa** | ratio < 0.68 | Il Dottore è "avaro" |
| **Media** | 0.68 ≤ ratio ≤ 0.87 | Offerta nella norma |
| **Alta** | ratio > 0.87 | Il Dottore è "generoso" |

| Modello | Accuracy test | CV media | CV std |
|---------|:---:|:---:|:---:|
| Decision Tree | 0.598 | 0.564 | 0.104 |
| **Random Forest** | 0.577 | **0.622** | **0.084** |
| Naive Bayes | 0.556 | 0.561 | 0.059 |

> Baseline casuale (3 classi): 33%. I modelli raddoppiano la performance.

### Regressione — stima in euro

Il modello di regressione (Random Forest Regressor) predice il ratio come valore continuo e lo converte in euro:

```
offerta_stimata = ratio_predetto × valore_atteso_premi_rimasti
```

| Metrica | Valore |
|---------|--------|
| MAE (euro) | **3.163 €** — errore medio assoluto |
| RMSE (euro) | 4.494 € |
| R² | 0.514 |

Il caso più accurato: offerta reale **8.000 €**, stima del modello **7.891 €** — errore di soli **109 €**.

### Le feature

Calcolate dinamicamente ricostruendo lo stato della partita dal JSON:

| Feature | Descrizione |
|---------|-------------|
| `valore_atteso` | Media dei premi rimasti — feature più predittiva |
| `max_rimasto` | Premio massimo ancora in gioco |
| `std_rimasti` | Deviazione standard dei premi rimasti |
| `n_pacchi_rimasti` | Pacchi ancora in gioco |
| `mediana_rimasti` | Mediana dei premi rimasti |
| `min_rimasto` | Premio minimo rimasto |
| `n_valori_alti` | Pacchi con valore ≥ 50.000 € |
| `num_offerta` | Turno dell'offerta (1–7) |

---

## 🎮 Il Dottore Artificiale — gioco da terminale

Il modello non è solo un numero su una tabella. `gioca.py` è un simulatore di *Affari Tuoi* da terminale che usa il modello addestrato come cervello del Dottore: ogni partita è generata casualmente, e ad ogni turno il Dottore artificiale calcola le stesse feature usate in training per stimare quanto offrirebbe quello reale.

```bash
# Prima genera il modello (una volta sola)
cd python
python3 affari_tuoi_ml.py

# Poi gioca
cd ../game
python3 gioca.py
```

Il gioco mostra il tabellone colorato, i premi rimasti divisi per fascia, il valore atteso matematico, il ratio stimato e l'offerta in euro. Alla fine confronta la scelta del giocatore con il valore reale del pacco.

---

## ⚙️ Come eseguire il codice ML

### Requisiti
```bash
pip install pandas scikit-learn matplotlib seaborn joblib
```

### Esecuzione
```bash
cd python
# Copia prima il JSON nella stessa cartella:
cp ../data/azioni_partite.json .
python3 affari_tuoi_ml.py
```

Genera automaticamente tutti i grafici, i CSV dei risultati e il modello serializzato in `game/`.

---

## 📚 Dataset

I dati provengono dal repository open-source [luzo7/affari-tuoi](https://github.com/luzo7/affari-tuoi), che raccoglie manualmente le puntate del programma in formato JSON. Per il contesto sull'analisi di fairness si è fatto riferimento a [PietroParenti/affari-tuoi-fairness](https://github.com/PietroParenti/affari-tuoi-fairness).

---

## 🛠️ Strumenti utilizzati

- **Python 3.9** — pandas, scikit-learn, matplotlib, seaborn, joblib
- **KNIME Analytics Platform 5** — replica visuale della pipeline ML
- **LaTeX** — stesura della tesina

---

## 📄 Licenza

Progetto a scopo accademico. I dati appartengono ai rispettivi autori dei repository citati.
