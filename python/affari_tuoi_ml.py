"""
==============================================================
  Progetto: Predizione delle offerte del Dottore — Affari Tuoi
  Corso:    Elementi di Intelligenza Artificiale
  Dataset:  github.com/luzo7/affari-tuoi (89 puntate, 2024)
==============================================================

Struttura dello script:
  1. Estrazione e feature engineering dal JSON
  2. Pre-processing (target categorico, normalizzazione, split)
  3. Addestramento e cross-validation (DT, RF, NB)
  4. Valutazione (accuracy, precision, recall, F1, conf. matrix)
  5. Interpretazione narrativa su esempi reali
  6. Salvataggio grafici e CSV dei risultati

Requisiti:
  pip install pandas scikit-learn matplotlib seaborn
"""

# ==============================================================
# 0. IMPORT
# ==============================================================
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)

# Cartella output per grafici e CSV
os.makedirs("output", exist_ok=True)

# Seme per riproducibilità
SEED = 42

# ==============================================================
# 1. ESTRAZIONE DAL JSON E FEATURE ENGINEERING
# ==============================================================

# Tutti i valori presenti in ogni puntata di Affari Tuoi
TUTTI_VALORI = [
    0, 1, 5, 10, 20, 50, 75, 100, 200, 500,
    1000, 5000, 10000, 15000, 20000, 30000,
    50000, 75000, 100000, 200000, 300000
]

print("=" * 60)
print("1. ESTRAZIONE DAL JSON E FEATURE ENGINEERING")
print("=" * 60)

# Percorso del file JSON — modifica se necessario
JSON_PATH = "azioni_partite.json"

with open(JSON_PATH, "r", encoding="utf-8") as f:
    dati_raw = json.load(f)

print(f"Puntate caricate: {len(dati_raw)}")

righe = []

for partita in dati_raw:
    data_puntata   = partita["data"]
    valori_rimasti = list(TUTTI_VALORI)   # copia fresca per ogni puntata
    num_offerta    = 0

    for azione in partita["azioni"]:
        tipo = azione["tipoAzione"]

        # Aggiorniamo lo stato: ogni apertura rimuove un valore dal pool
        if tipo == "Apertura":
            val_aperto = azione["args"]["valPaccoAperto"]
            if val_aperto in valori_rimasti:
                valori_rimasti.remove(val_aperto)

        # Quando c'è un'offerta, calcoliamo le feature sullo stato attuale
        elif tipo == "Offerta":
            num_offerta  += 1
            val_offerta   = azione["args"]["valOfferta"]
            stato_offerta = azione["args"]["statoOfferta"]
            n             = len(valori_rimasti)

            if n == 0:
                continue  # caso degenere, saltiamo

            # --- Feature numeriche ---
            valore_atteso  = np.mean(valori_rimasti)
            mediana        = np.median(valori_rimasti)
            massimo        = max(valori_rimasti)
            minimo         = min(valori_rimasti)
            std            = np.std(valori_rimasti)
            n_alti         = sum(1 for v in valori_rimasti if v >= 50_000)

            # ratio = quanto l'offerta copre del valore atteso (es. 0.79 = 79%)
            ratio = val_offerta / valore_atteso if valore_atteso > 0 else 0

            righe.append({
                "data"            : data_puntata,
                "num_offerta"     : num_offerta,
                "val_offerta"     : val_offerta,
                "stato_offerta"   : stato_offerta,
                "n_pacchi_rimasti": n,
                "valore_atteso"   : round(valore_atteso, 2),
                "mediana_rimasti" : mediana,
                "max_rimasto"     : massimo,
                "min_rimasto"     : minimo,
                "std_rimasti"     : round(std, 2),
                "n_valori_alti"   : n_alti,
                "ratio_offerta_va": round(ratio, 4),
            })

df = pd.DataFrame(righe)

print(f"Offerte estratte: {df.shape[0]}")
print(f"Feature disponibili: {df.shape[1]}")
print(f"Valori nulli: {df.isnull().sum().sum()}")

# Salva il dataset grezzo per KNIME
df.to_csv("output/dataset_offerte.csv", index=False)
print("-> Salvato: output/dataset_offerte.csv")

# ==============================================================
# 2. PRE-PROCESSING
# ==============================================================

print("\n" + "=" * 60)
print("2. PRE-PROCESSING")
print("=" * 60)

# --- 2a. Costruzione del target categorico ---
# Soglie calibrate sulla distribuzione reale (p32 ~ 0.68, p70 ~ 0.87)
# Distribuzione risultante: Bassa 32% | Media 38% | Alta 30%
SOGLIA_BASSA = 0.68
SOGLIA_ALTA  = 0.87

def classifica_offerta(ratio):
    if ratio < SOGLIA_BASSA:
        return "Bassa"
    elif ratio <= SOGLIA_ALTA:
        return "Media"
    else:
        return "Alta"

df["classe_offerta"] = df["ratio_offerta_va"].apply(classifica_offerta)

print(f"Soglie: ratio < {SOGLIA_BASSA} -> Bassa | {SOGLIA_BASSA}-{SOGLIA_ALTA} -> Media | > {SOGLIA_ALTA} -> Alta")
print(f"\nDistribuzione classi:\n{df['classe_offerta'].value_counts()}")
print(f"\nDistribuzione percentuale:")
print((df['classe_offerta'].value_counts(normalize=True) * 100).round(1).to_string())

# Encoding del target: Alta=0, Bassa=1, Media=2 (ordine alfabetico LabelEncoder)
le = LabelEncoder()
df["target"] = le.fit_transform(df["classe_offerta"])
print(f"\nMapping classi: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# --- 2b. Selezione feature di input ---
FEATURE_COLS = [
    "num_offerta",
    "n_pacchi_rimasti",
    "valore_atteso",
    "mediana_rimasti",
    "max_rimasto",
    "min_rimasto",
    "std_rimasti",
    "n_valori_alti",
]

X = df[FEATURE_COLS].copy()
y = df["target"].copy()

print(f"\nFeature di input ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"Shape X: {X.shape} | Shape y: {y.shape}")

# --- 2c. Suddivisione train/test (75% train, 25% test) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=SEED, stratify=y
)

print(f"\nTrain: {X_train.shape[0]} campioni | Test: {X_test.shape[0]} campioni")
print(f"Distribuzione classi nel train:\n{pd.Series(y_train).value_counts().sort_index()}")

# Nota: la normalizzazione MinMaxScaler e' inclusa nelle pipeline dei modelli
# per evitare data leakage tra train e test

# ==============================================================
# 3. DEFINIZIONE DEI MODELLI E CROSS-VALIDATION
# ==============================================================

print("\n" + "=" * 60)
print("3. CROSS-VALIDATION (KFold k=10)")
print("=" * 60)

# Ogni modello e' avvolto in una pipeline: scaler + classificatore
pipelines = {
    "Decision Tree" : make_pipeline(
        MinMaxScaler(),
        DecisionTreeClassifier(random_state=SEED, max_depth=5)
    ),
    "Random Forest" : make_pipeline(
        MinMaxScaler(),
        RandomForestClassifier(n_estimators=100, random_state=SEED)
    ),
    "Naive Bayes"   : make_pipeline(
        MinMaxScaler(),
        GaussianNB()
    ),
}

kfold = KFold(n_splits=10, shuffle=True, random_state=SEED)

risultati_cv = {}
for nome, pipeline in pipelines.items():
    scores = cross_val_score(pipeline, X_train, y_train, cv=kfold, scoring="accuracy")
    risultati_cv[nome] = scores
    print(f"\n{nome}")
    print(f"  Scores CV: {scores.round(3)}")
    print(f"  Media: {scores.mean():.3f} | Std: {scores.std():.3f}")

# ==============================================================
# 4. ADDESTRAMENTO FINALE E VALUTAZIONE SUL TEST SET
# ==============================================================

print("\n" + "=" * 60)
print("4. VALUTAZIONE SUL TEST SET")
print("=" * 60)

nomi_classi = le.classes_          # ["Alta", "Bassa", "Media"]
risultati_test = {}

for nome, pipeline in pipelines.items():
    # Addestriamo sul train set completo
    pipeline.fit(X_train, y_train)

    # Predizione sul test set
    y_pred = pipeline.predict(X_test)

    # Metriche
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                   target_names=nomi_classi,
                                   output_dict=True)

    risultati_test[nome] = {
        "accuracy"  : acc,
        "y_pred"    : y_pred,
        "report"    : report,
        "pipeline"  : pipeline,
    }

    print(f"\n{'---'*14}")
    print(f"  {nome}")
    print(f"{'---'*14}")
    print(f"  Accuracy sul test set: {acc:.3f}")
    print(classification_report(y_test, y_pred, target_names=nomi_classi))

# ==============================================================
# 5. GRAFICI
# ==============================================================

print("\n" + "=" * 60)
print("5. GENERAZIONE GRAFICI")
print("=" * 60)

PALETTE = {"Decision Tree": "#378ADD", "Random Forest": "#1D9E75", "Naive Bayes": "#BA7517"}
nomi = list(pipelines.keys())

# --- Grafico 1: Confronto accuracy CV vs Test ---
fig, ax = plt.subplots(figsize=(8, 4))
acc_cv   = [risultati_cv[n].mean() for n in nomi]
acc_test = [risultati_test[n]["accuracy"] for n in nomi]
x = np.arange(len(nomi))
w = 0.35

bars1 = ax.bar(x - w/2, acc_cv,   w, label="Cross-validation (media)", color=[PALETTE[n] for n in nomi], alpha=0.6)
bars2 = ax.bar(x + w/2, acc_test, w, label="Test set",                  color=[PALETTE[n] for n in nomi])

ax.set_ylabel("Accuracy")
ax.set_title("Confronto accuracy: cross-validation vs test set")
ax.set_xticks(x)
ax.set_xticklabels(nomi)
ax.set_ylim(0, 1)
ax.legend()
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("output/grafico_accuracy.png", dpi=150, bbox_inches="tight")
plt.close()
print("-> Salvato: output/grafico_accuracy.png")

# --- Grafico 2: Matrici di confusione (3 affiancate) ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, nome in zip(axes, nomi):
    y_pred = risultati_test[nome]["y_pred"]
    cm     = confusion_matrix(y_test, y_pred)
    disp   = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nomi_classi)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(nome, fontsize=11)

plt.suptitle("Matrici di confusione - Test set", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("output/matrici_confusione.png", dpi=150, bbox_inches="tight")
plt.close()
print("-> Salvato: output/matrici_confusione.png")

# --- Grafico 3: Feature importance Random Forest ---
rf_pipeline = risultati_test["Random Forest"]["pipeline"]
rf_model    = rf_pipeline.named_steps["randomforestclassifier"]
importances = rf_model.feature_importances_
feat_imp    = pd.Series(importances, index=FEATURE_COLS).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.barh(feat_imp.index, feat_imp.values, color="#1D9E75")
ax.set_xlabel("Importanza")
ax.set_title("Feature importance - Random Forest")
for bar, val in zip(bars, feat_imp.values):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("output/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("-> Salvato: output/feature_importance.png")

# --- Grafico 4: Boxplot ratio per turno ---
fig, ax = plt.subplots(figsize=(8, 4))
turni        = sorted(df["num_offerta"].unique())
data_boxplot = [df[df["num_offerta"] == t]["ratio_offerta_va"].values for t in turni]
bp = ax.boxplot(data_boxplot, labels=[f"Turno {t}" for t in turni], patch_artist=True)
for patch in bp["boxes"]:
    patch.set_facecolor("#B5D4F4")
ax.axhline(y=SOGLIA_BASSA, color="#E24B4A", linestyle="--", linewidth=1,
           label=f"Soglia bassa ({SOGLIA_BASSA})")
ax.axhline(y=SOGLIA_ALTA,  color="#639922", linestyle="--", linewidth=1,
           label=f"Soglia alta ({SOGLIA_ALTA})")
ax.set_ylabel("Ratio offerta / valore atteso")
ax.set_title("Comportamento del Dottore per turno di offerta")
ax.legend()
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("output/ratio_per_turno.png", dpi=150, bbox_inches="tight")
plt.close()
print("-> Salvato: output/ratio_per_turno.png")

# --- Grafico 5: Distribuzione classi ---
fig, ax = plt.subplots(figsize=(5, 4))
counts = df["classe_offerta"].value_counts()
colors = ["#9FE1CB", "#E24B4A", "#B5D4F4"]   # Alta, Bassa, Media
ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
       colors=colors, startangle=90,
       wedgeprops={"edgecolor": "white", "linewidth": 1.5})
ax.set_title("Distribuzione classi offerta nel dataset")
plt.tight_layout()
plt.savefig("output/distribuzione_classi.png", dpi=150, bbox_inches="tight")
plt.close()
print("-> Salvato: output/distribuzione_classi.png")

# ==============================================================
# 6. SALVATAGGIO RISULTATI IN CSV
# ==============================================================

print("\n" + "=" * 60)
print("6. SALVATAGGIO RISULTATI")
print("=" * 60)

righe_metriche = []
for nome in nomi:
    r   = risultati_test[nome]["report"]
    row = {
        "Modello"  : nome,
        "Accuracy" : round(risultati_test[nome]["accuracy"], 4),
        "CV_media" : round(risultati_cv[nome].mean(), 4),
        "CV_std"   : round(risultati_cv[nome].std(), 4),
    }
    for classe in nomi_classi:
        row[f"Precision_{classe}"] = round(r[classe]["precision"], 4)
        row[f"Recall_{classe}"]    = round(r[classe]["recall"], 4)
        row[f"F1_{classe}"]        = round(r[classe]["f1-score"], 4)
    righe_metriche.append(row)

df_metriche = pd.DataFrame(righe_metriche)
df_metriche.to_csv("output/risultati_modelli.csv", index=False)
print(f"-> Salvato: output/risultati_modelli.csv")
print(f"\nTabella riassuntiva:\n{df_metriche[['Modello','Accuracy','CV_media','CV_std']].to_string(index=False)}")

# ==============================================================
# 7. INTERPRETAZIONE NARRATIVA
# ==============================================================

print("\n" + "=" * 60)
print("7. INTERPRETAZIONE NARRATIVA - esempi reali dal test set")
print("=" * 60)

rf_pipeline.fit(X_train, y_train)
df_test               = df.iloc[X_test.index].copy()
df_test["pred_classe"]  = le.inverse_transform(risultati_test["Random Forest"]["y_pred"])
df_test["reale_classe"] = le.inverse_transform(y_test.values)
df_test["corretta"]     = df_test["pred_classe"] == df_test["reale_classe"]
df_test["differenza_euro"] = (df_test["valore_atteso"] - df_test["val_offerta"]).round(0)

print(f"\nAccuracy RF sul test set: {df_test['corretta'].mean():.1%}")

print(f"\nEsempi con offerta BASSA predetta correttamente:")
for _, row in df_test[(df_test["reale_classe"]=="Bassa") & df_test["corretta"]].head(3).iterrows():
    print(f"  {row['data']} | Turno {int(row['num_offerta'])} | VA: EUR{row['valore_atteso']:,.0f} "
          f"| Offerta: EUR{int(row['val_offerta']):,} "
          f"| Trattenuto: EUR{row['differenza_euro']:,.0f} ({(1-row['ratio_offerta_va'])*100:.0f}%)")

print(f"\nEsempi con offerta ALTA predetta correttamente:")
for _, row in df_test[(df_test["reale_classe"]=="Alta") & df_test["corretta"]].head(3).iterrows():
    print(f"  {row['data']} | Turno {int(row['num_offerta'])} | VA: EUR{row['valore_atteso']:,.0f} "
          f"| Offerta: EUR{int(row['val_offerta']):,} "
          f"| Ratio: {row['ratio_offerta_va']:.2f}")

print("\n" + "=" * 60)
print("8. REGRESSIONE - stima dell'offerta in euro")
print("=" * 60)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Target continuo: il ratio offerta/valore atteso
y_reg = df["ratio_offerta_va"].copy()

# Stessi split con stesso seed -> stessi indici di train/test
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_reg, test_size=0.25, random_state=SEED
)

# Pipeline regressione
reg_pipeline = make_pipeline(
    MinMaxScaler(),
    RandomForestRegressor(n_estimators=100, random_state=SEED)
)

# Cross-validation su MAE
from sklearn.model_selection import cross_val_score
cv_mae = cross_val_score(reg_pipeline, X_train_r, y_train_r,
                          cv=KFold(n_splits=10, shuffle=True, random_state=SEED),
                          scoring="neg_mean_absolute_error")
print(f"Cross-validation MAE (ratio): {(-cv_mae.mean()):.4f} ± {cv_mae.std():.4f}")

# Addestramento e predizione
reg_pipeline.fit(X_train_r, y_train_r)
y_pred_ratio = reg_pipeline.predict(X_test_r)

# Metriche sul ratio
mae  = mean_absolute_error(y_test_r, y_pred_ratio)
rmse = mean_squared_error(y_test_r, y_pred_ratio) ** 0.5
r2   = r2_score(y_test_r, y_pred_ratio)

print(f"\nMetriche sul ratio predetto (test set):")
print(f"  MAE  (ratio): {mae:.4f}  → in media ±{mae*100:.1f}% dal ratio reale")
print(f"  RMSE (ratio): {rmse:.4f}")
print(f"  R²:           {r2:.4f}")

# Conversione in euro: offerta_stimata = ratio_predetto * valore_atteso
df_reg = df.iloc[X_test_r.index].copy()
df_reg["ratio_predetto"]   = y_pred_ratio
df_reg["offerta_stimata"]  = (df_reg["ratio_predetto"] * df_reg["valore_atteso"]).round(0)
df_reg["errore_euro"]      = (df_reg["offerta_stimata"] - df_reg["val_offerta"]).round(0)
df_reg["errore_abs_euro"]  = df_reg["errore_euro"].abs()

mae_euro  = df_reg["errore_abs_euro"].mean()
rmse_euro = (df_reg["errore_euro"]**2).mean()**0.5

print(f"\nMetriche sull'offerta stimata in euro:")
print(f"  MAE  (euro): EUR{mae_euro:,.0f}  → errore medio assoluto")
print(f"  RMSE (euro): EUR{rmse_euro:,.0f}")

# Salva risultati regressione
df_reg[["data","num_offerta","valore_atteso","val_offerta",
        "ratio_offerta_va","ratio_predetto",
        "offerta_stimata","errore_euro"]].to_csv(
    "output/risultati_regressione.csv", index=False)
print("-> Salvato: output/risultati_regressione.csv")

# Grafico: offerta reale vs offerta stimata
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(df_reg["val_offerta"], df_reg["offerta_stimata"],
           alpha=0.6, color="#1D9E75", edgecolors="white", linewidth=0.5, s=50)
lim = max(df_reg["val_offerta"].max(), df_reg["offerta_stimata"].max()) * 1.05
ax.plot([0, lim], [0, lim], "--", color="#E24B4A", linewidth=1.2, label="Predizione perfetta")
ax.set_xlabel("Offerta reale del Dottore (€)")
ax.set_ylabel("Offerta stimata dal modello (€)")
ax.set_title("Regressione: offerta reale vs offerta stimata")
ax.legend()
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
# Formatta assi in migliaia
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"€{x/1000:.0f}k"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"€{x/1000:.0f}k"))
plt.tight_layout()
plt.savefig("output/regressione_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("-> Salvato: output/regressione_scatter.png")

# Grafico: distribuzione errori in euro
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(df_reg["errore_euro"], bins=25, color="#378ADD",
        edgecolor="white", linewidth=0.5)
ax.axvline(0, color="#E24B4A", linestyle="--", linewidth=1.2, label="Errore zero")
ax.axvline(mae_euro, color="#BA7517", linestyle="--", linewidth=1,
           label=f"MAE = €{mae_euro:,.0f}")
ax.axvline(-mae_euro, color="#BA7517", linestyle="--", linewidth=1)
ax.set_xlabel("Errore (offerta stimata − offerta reale) in €")
ax.set_ylabel("Frequenza")
ax.set_title("Distribuzione degli errori di stima in euro")
ax.legend()
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("output/regressione_errori.png", dpi=150, bbox_inches="tight")
plt.close()
print("-> Salvato: output/regressione_errori.png")

# Esempi narrativi: casi più accurati
print(f"\nI 5 casi in cui il modello ha stimato meglio l'offerta:")
for _, row in df_reg.nsmallest(5, "errore_abs_euro").iterrows():
    print(f"  {row['data']} | Turno {int(row['num_offerta'])} "
          f"| VA: EUR{row['valore_atteso']:,.0f} "
          f"| Reale: EUR{int(row['val_offerta']):,} "
          f"| Stimata: EUR{int(row['offerta_stimata']):,} "
          f"| Errore: EUR{int(row['errore_euro']):+,}")

print(f"\nI 3 casi con errore maggiore:")
for _, row in df_reg.nlargest(3, "errore_abs_euro").iterrows():
    print(f"  {row['data']} | Turno {int(row['num_offerta'])} "
          f"| VA: EUR{row['valore_atteso']:,.0f} "
          f"| Reale: EUR{int(row['val_offerta']):,} "
          f"| Stimata: EUR{int(row['offerta_stimata']):,} "
          f"| Errore: EUR{int(row['errore_euro']):+,}")

# ==============================================================
# 9. SALVATAGGIO DEL MODELLO PER IL GIOCO
# ==============================================================

print("\n" + "=" * 60)
print("9. SALVATAGGIO MODELLO PER IL GIOCO")
print("=" * 60)

import joblib
import os

os.makedirs("../game", exist_ok=True)

# Salviamo il regressore addestrato su TUTTI i dati (non solo train)
reg_finale = make_pipeline(
    MinMaxScaler(),
    RandomForestRegressor(n_estimators=100, random_state=SEED)
)
reg_finale.fit(X, y_reg)   # addestrato su tutto il dataset
joblib.dump(reg_finale, "../game/modello_dottore.pkl")
print("-> Salvato: ../game/modello_dottore.pkl")

# Salviamo anche la lista delle feature nell'ordine giusto
joblib.dump(FEATURE_COLS, "../game/feature_cols.pkl")
print("-> Salvato: ../game/feature_cols.pkl")

print("\n" + "=" * 60)
print("Script completato. File salvati nella cartella output/")
print("=" * 60)
