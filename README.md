# 🎯 Il Dottore sotto esame — Machine Learning applicato ad *Affari Tuoi*

> Progetto per il corso di **Elementi di Intelligenza Artificiale**  
> Università degli Studi di Napoli Federico II — A.A. 2025/2026  
> Prof. Giancarlo Sperlì

---

## 📌 Descrizione

*Affari Tuoi* è il programma televisivo più visto d'Italia. Al centro dello show c'è il **Dottore**, che propone offerte in denaro al concorrente in cambio del suo pacco. Ma le sue offerte sono eque?

Questo progetto risponde a questa domanda con gli strumenti del **Machine Learning supervisionato**: a partire da 89 puntate reali della stagione 2023/2024, vengono addestrati tre classificatori in grado di predire se l'offerta del Dottore è **bassa**, **media** o **alta** rispetto al valore matematicamente atteso.

L'analisi rivela che le offerte del Dottore **non sono casuali** — seguono pattern statisticamente riconoscibili legati allo stato della partita, al turno di offerta e ai premi rimasti in gioco.

---

## 🗂️ Struttura del repository

```
AffariTuoi/
│
├── data/
│   └── azioni_partite.json        # Dataset originale (89 puntate, fonte: github.com/luzo7/affari-tuoi)
│
├── python/
│   ├── affari_tuoi_ml.py          # Script principale
│   └── output/
│       ├── dataset_offerte.csv    # Dataset elaborato (386 offerte con feature calcolate)
│       ├── risultati_modelli.csv  # Accuracy, precision, recall, F1 per ogni modello
│       ├── grafico_accuracy.png
│       ├── matrici_confusione.png
│       ├── feature_importance.png
│       ├── ratio_per_turno.png
│       └── distribuzione_classi.png
│
├── knime/
│   └── AffariTuoi/                # Workflow KNIME 5 (Random Forest Learner + Scorer)
│
├── latex/
│   ├── tesina_affari_tuoi.tex     # Sorgente LaTeX
│   ├── tesina_affari_tuoi.pdf     # Tesina compilata
│   └── img/                       # Grafici e screenshot usati nella tesina
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

Un ratio = 1 significa offerta perfettamente equa. Il dataset mostra una media di **0.789** — il Dottore trattiene in media il 21% del valore atteso.

### Il target
Il ratio viene discretizzato in 3 classi bilanciate:

| Classe | Soglia | Significato |
|--------|--------|-------------|
| **Bassa** | ratio < 0.68 | Il Dottore è "avaro" |
| **Media** | 0.68 ≤ ratio ≤ 0.87 | Offerta nella norma |
| **Alta** | ratio > 0.87 | Il Dottore è "generoso" |

### Le feature
Calcolate dinamicamente per ogni offerta ricostruendo lo stato della partita dal JSON:

- `valore_atteso` — media dei premi rimasti
- `n_pacchi_rimasti` — pacchi ancora in gioco
- `mediana_rimasti`, `max_rimasto`, `min_rimasto`, `std_rimasti`
- `n_valori_alti` — pacchi con valore ≥ €50.000
- `num_offerta` — turno dell'offerta (1–7)

### I modelli
| Modello | Accuracy test | CV media | CV std |
|---------|:---:|:---:|:---:|
| Decision Tree | 0.598 | 0.564 | 0.104 |
| **Random Forest** | 0.577 | **0.622** | **0.084** |
| Naive Bayes | 0.556 | 0.561 | 0.059 |

> Baseline casuale (3 classi): 33%. I modelli raddoppiano la performance.  
> La Random Forest mostra la maggiore **stabilità** in cross-validation (std più bassa).

---

## 📊 Risultati principali

- Il Dottore **non è casuale**: l'accuracy sistematicamente sopra il 33% lo dimostra
- È più aggressivo al **3° turno** (ratio medio 0.729) e più generoso nelle fasi finali (0.893)
- Le feature più predittive sono `valore_atteso`, `max_rimasto` e `std_rimasti`
- La classe **Media** è la più difficile da classificare — il Dottore è meno prevedibile nelle offerte "nella norma"

---

## ⚙️ Come eseguire il codice

### Requisiti
```bash
pip install pandas scikit-learn matplotlib seaborn
```

### Esecuzione
```bash
# Metti affari_tuoi_ml.py e azioni_partite.json nella stessa cartella, poi:
python affari_tuoi_ml.py
```

I risultati vengono salvati nella cartella `output/` che viene creata automaticamente.

---

## 📚 Dataset

I dati provengono dal repository open-source [luzo7/affari-tuoi](https://github.com/luzo7/affari-tuoi), che raccoglie manualmente le puntate della stagione 2023/2024 in formato JSON. Per l'analisi statistica di fairness si è fatto riferimento anche a [PietroParenti/affari-tuoi-fairness](https://github.com/PietroParenti/affari-tuoi-fairness).

---

## 🛠️ Strumenti utilizzati

- **Python 3** con scikit-learn, pandas, matplotlib
- **KNIME Analytics Platform 5** per la replica visuale della pipeline
- **LaTeX** per la stesura della tesina

---

## 📄 Licenza

Progetto a scopo accademico. I dati appartengono ai rispettivi autori dei repository citati.
