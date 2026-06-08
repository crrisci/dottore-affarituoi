"""
==============================================================
  Affari Tuoi — Il Dottore Artificiale
  Simulatore da terminale con modello ML reale
==============================================================

Come funziona:
  1. I 21 premi vengono distribuiti casualmente nei pacchi
  2. Scegli il tuo pacco
  3. Ad ogni turno apri i pacchi degli altri concorrenti
  4. Il Dottore (il nostro modello ML) ti fa un'offerta
  5. Decidi se accettare o continuare
  6. Alla fine scopri cosa c'era nel tuo pacco

Requisiti:
  - aver eseguito affari_tuoi_ml.py almeno una volta
    (genera modello_dottore.pkl e feature_cols.pkl)
  - pip install joblib scikit-learn numpy
"""

import random
import numpy as np
import joblib
import os
import sys

# ── Colori ANSI per il terminale ───────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
GIALLO = "\033[93m"
VERDE  = "\033[92m"
ROSSO  = "\033[91m"
CIANO  = "\033[96m"
GRIGIO = "\033[90m"
BLU    = "\033[94m"

# ── Premi ufficiali di Affari Tuoi ─────────────────────────────
TUTTI_VALORI = [
    0, 1, 5, 10, 20, 50, 75, 100, 200, 500,
    1000, 5000, 10000, 15000, 20000, 30000,
    50000, 75000, 100000, 200000, 300000
]

# ── Turni: quanti pacchi aprire per ogni turno ─────────────────
PACCHI_PER_TURNO = [6, 3, 3, 2, 2, 1, 1, 1, 1]

def formatta_euro(valore):
    """Formatta un valore in euro con separatore migliaia."""
    if valore == 0:
        return "0 €"
    return f"{valore:,} €".replace(",", ".")

def colore_valore(valore):
    """Colora il valore in base all'entità."""
    if valore >= 50000:
        return f"{ROSSO}{BOLD}{formatta_euro(valore)}{RESET}"
    elif valore >= 5000:
        return f"{GIALLO}{formatta_euro(valore)}{RESET}"
    elif valore >= 100:
        return f"{VERDE}{formatta_euro(valore)}{RESET}"
    else:
        return f"{GRIGIO}{formatta_euro(valore)}{RESET}"

def stampa_intestazione():
    print(f"\n{BLU}{'═'*56}{RESET}")
    print(f"{BLU}{BOLD}{'AFFARI TUOI — IL DOTTORE ARTIFICIALE':^56}{RESET}")
    print(f"{BLU}{'═'*56}{RESET}")
    print(f"{GRIGIO}  Il Dottore è un modello ML addestrato su 89 puntate{RESET}")
    print(f"{GRIGIO}  reali della stagione 2023/2024.{RESET}")
    print(f"{BLU}{'─'*56}{RESET}\n")

def stampa_tabellone(pacchi, pacco_giocatore, aperti):
    """Stampa il tabellone con i pacchi disponibili."""
    print(f"\n{CIANO}{'─'*56}{RESET}")
    print(f"{CIANO}{BOLD}  TABELLONE{RESET}")
    print(f"{CIANO}{'─'*56}{RESET}")
    riga = ""
    for n in range(1, 22):
        if n in aperti:
            riga += f"  {GRIGIO}[✗]{RESET}"
        elif n == pacco_giocatore:
            riga += f"  {GIALLO}{BOLD}[{n:2d}]{RESET}"
        else:
            riga += f"  {VERDE}[{n:2d}]{RESET}"
        if n % 7 == 0:
            print(riga)
            riga = ""
    if riga:
        print(riga)
    print(f"\n  {GIALLO}{BOLD}[##]{RESET} = il tuo pacco   "
          f"{VERDE}[nn]{RESET} = disponibile   "
          f"{GRIGIO}[✗]{RESET} = aperto")

def stampa_valori_rimasti(valori_rimasti):
    """Stampa i premi ancora in gioco divisi in alti e bassi."""
    print(f"\n{CIANO}  Premi ancora in gioco:{RESET}")
    bassi  = [v for v in valori_rimasti if v < 5000]
    medi   = [v for v in valori_rimasti if 5000 <= v < 50000]
    alti   = [v for v in valori_rimasti if v >= 50000]
    if alti:
        print(f"  {ROSSO}Alti:{RESET}  " +
              "  ".join(colore_valore(v) for v in sorted(alti, reverse=True)))
    if medi:
        print(f"  {GIALLO}Medi:{RESET}  " +
              "  ".join(colore_valore(v) for v in sorted(medi, reverse=True)))
    if bassi:
        print(f"  {GRIGIO}Bassi:{RESET} " +
              "  ".join(colore_valore(v) for v in sorted(bassi, reverse=True)))

def calcola_feature(valori_rimasti, num_offerta):
    """Calcola le feature esattamente come durante il training."""
    n  = len(valori_rimasti)
    va = np.mean(valori_rimasti)
    return {
        "num_offerta"      : num_offerta,
        "n_pacchi_rimasti" : n,
        "valore_atteso"    : va,
        "mediana_rimasti"  : np.median(valori_rimasti),
        "max_rimasto"      : max(valori_rimasti),
        "min_rimasto"      : min(valori_rimasti),
        "std_rimasti"      : np.std(valori_rimasti),
        "n_valori_alti"    : sum(1 for v in valori_rimasti if v >= 50000),
    }

def offerta_dottore(modello, feature_cols, valori_rimasti, num_offerta):
    """Usa il modello ML per stimare l'offerta del Dottore."""
    feat = calcola_feature(valori_rimasti, num_offerta)
    X    = np.array([[feat[c] for c in feature_cols]])
    ratio_pred   = modello.predict(X)[0]
    ratio_pred   = max(0.1, min(ratio_pred, 1.5))  # clamp ragionevole
    va           = feat["valore_atteso"]
    offerta_raw  = ratio_pred * va
    # Arrotondiamo all'intero più "bello" (multiplo di 500 o 1000)
    if offerta_raw >= 10000:
        offerta = round(offerta_raw / 1000) * 1000
    elif offerta_raw >= 1000:
        offerta = round(offerta_raw / 500) * 500
    else:
        offerta = round(offerta_raw / 50) * 50
    return int(offerta), ratio_pred, va

def input_numero(prompt, validi):
    """Richiede un numero tra quelli validi."""
    while True:
        try:
            scelta = int(input(prompt))
            if scelta in validi:
                return scelta
            print(f"  {ROSSO}Scegli un numero tra: {sorted(validi)}{RESET}")
        except ValueError:
            print(f"  {ROSSO}Inserisci un numero valido.{RESET}")

def input_si_no(prompt):
    """Richiede sì o no."""
    while True:
        r = input(prompt).strip().lower()
        if r in ("s", "si", "sì", "y", "yes"):
            return True
        if r in ("n", "no"):
            return False
        print(f"  {ROSSO}Rispondi con S (sì) o N (no).{RESET}")

def gioca():
    # ── Carica il modello ───────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path_modello = os.path.join(script_dir, "modello_dottore.pkl")
    path_feature = os.path.join(script_dir, "feature_cols.pkl")

    if not os.path.exists(path_modello):
        print(f"{ROSSO}ERRORE: modello_dottore.pkl non trovato.{RESET}")
        print("Esegui prima: python3 ../python/affari_tuoi_ml.py")
        sys.exit(1)

    modello      = joblib.load(path_modello)
    feature_cols = joblib.load(path_feature)

    stampa_intestazione()
    input(f"  Premi {BOLD}INVIO{RESET} per iniziare la partita...\n")

    # ── Setup partita ───────────────────────────────────────────
    valori = TUTTI_VALORI[:]
    random.shuffle(valori)
    pacchi = {n: v for n, v in enumerate(valori, 1)}   # {1: valore, ...}
    aperti = set()

    print(f"  I 21 premi sono stati distribuiti casualmente nei pacchi.\n")

    # ── Scelta del pacco ────────────────────────────────────────
    stampa_tabellone(pacchi, None, aperti)
    print(f"\n  {BOLD}Scegli il tuo pacco{RESET} (1-21):")
    pacco_giocatore = input_numero("  → Pacco numero: ", set(range(1, 22)))
    valori_rimasti  = [v for n, v in pacchi.items() if n != pacco_giocatore]

    print(f"\n  {GIALLO}{BOLD}Pacco {pacco_giocatore} scelto!{RESET} "
          f"Custodiscilo bene... 🎁\n")

    # ── Ciclo di gioco ──────────────────────────────────────────
    num_offerta   = 0
    fine_partita  = False
    esito         = None

    for turno, n_aprire in enumerate(PACCHI_PER_TURNO, 1):
        disponibili = [n for n in range(1, 22)
                       if n not in aperti and n != pacco_giocatore]
        if len(disponibili) == 0:
            break

        n_aprire = min(n_aprire, len(disponibili))

        print(f"\n{BLU}{'═'*56}{RESET}")
        print(f"{BLU}{BOLD}  TURNO {turno} — Apri {n_aprire} "
              f"pacco{'i' if n_aprire>1 else ''}{RESET}")
        print(f"{BLU}{'═'*56}{RESET}")

        stampa_tabellone(pacchi, pacco_giocatore, aperti)
        stampa_valori_rimasti(valori_rimasti)

        # Apertura pacchi
        for i in range(n_aprire):
            disponibili = [n for n in range(1, 22)
                           if n not in aperti and n != pacco_giocatore]
            print(f"\n  Apertura {i+1}/{n_aprire} — "
                  f"Pacchi disponibili: {sorted(disponibili)}")
            da_aprire = input_numero("  → Apri il pacco numero: ",
                                     set(disponibili))
            valore    = pacchi[da_aprire]
            aperti.add(da_aprire)
            valori_rimasti.remove(valore)

            print(f"\n  Pacco {da_aprire}: {colore_valore(valore)}  ", end="")
            if valore >= 50000:
                print(f"{ROSSO}Peccato! Un premio alto è uscito.{RESET}")
            elif valore == 0:
                print(f"{VERDE}Zero euro fuori! Ottimo!{RESET}")
            else:
                print()

        # ── Offerta del Dottore ─────────────────────────────────
        num_offerta += 1
        va = np.mean(valori_rimasti)

        offerta, ratio_pred, _ = offerta_dottore(
            modello, feature_cols, valori_rimasti, num_offerta)

        print(f"\n{CIANO}{'─'*56}{RESET}")
        print(f"{CIANO}{BOLD}  📞 IL DOTTORE CHIAMA...{RESET}")
        print(f"{CIANO}{'─'*56}{RESET}")
        print(f"\n  Valore atteso matematico: {colore_valore(int(va))}")
        print(f"  Ratio stimato dal modello: {ratio_pred:.2f} "
              f"({ratio_pred*100:.0f}% del valore atteso)\n")
        print(f"  {GIALLO}{BOLD}💰 OFFERTA DEL DOTTORE: "
              f"{formatta_euro(offerta)}{RESET}\n")

        # Mostra valori rimasti prima della decisione
        stampa_valori_rimasti(valori_rimasti)

        print(f"\n  Pacchi rimasti (escluso il tuo): {len(valori_rimasti)}")

        if len(valori_rimasti) == 1:
            # Ultimo pacco: cambio o tieni?
            print(f"\n  {BOLD}Vuoi cambiare il tuo pacco con quello rimasto?{RESET}")
            cambia = input_si_no("  → Cambia? (S/N): ")
            if cambia:
                altro = [n for n in range(1, 22)
                         if n not in aperti and n != pacco_giocatore][0]
                print(f"\n  Hai cambiato con il pacco {altro}!")
                pacco_giocatore, _ = altro, pacco_giocatore
            fine_partita = True
            esito = "pacco"
            break

        accetta = input_si_no(f"\n  → Accetti l'offerta di "
                              f"{formatta_euro(offerta)}? (S/N): ")

        if accetta:
            fine_partita = True
            esito        = "offerta"
            break

        print(f"\n  {VERDE}Hai rifiutato. Avanti con la partita!{RESET}")

    # ── Fine partita ────────────────────────────────────────────
    valore_pacco = pacchi[pacco_giocatore]

    print(f"\n{BLU}{'═'*56}{RESET}")
    print(f"{BLU}{BOLD}{'FINE PARTITA':^56}{RESET}")
    print(f"{BLU}{'═'*56}{RESET}\n")
    print(f"  Nel tuo pacco ({pacco_giocatore}) c'era: "
          f"{colore_valore(valore_pacco)}\n")

    if esito == "offerta":
        print(f"  Hai accettato l'offerta del Dottore: "
              f"{GIALLO}{BOLD}{formatta_euro(offerta)}{RESET}")
        diff = offerta - valore_pacco
        if diff > 0:
            print(f"  {VERDE}Scelta giusta! Hai guadagnato "
                  f"{formatta_euro(diff)} in più rispetto al pacco.{RESET}")
        elif diff < 0:
            print(f"  {ROSSO}Nel pacco c'era di più... "
                  f"avresti vinto {formatta_euro(abs(diff))} in più.{RESET}")
        else:
            print(f"  {VERDE}Offerta perfettamente allineata al pacco!{RESET}")
    else:
        print(f"  Hai aperto il tuo pacco e vinto: "
              f"{colore_valore(valore_pacco)}")

    print(f"\n{BLU}{'─'*56}{RESET}")
    print(f"  {GRIGIO}Statistiche partita:{RESET}")
    print(f"  Offerte ricevute: {num_offerta}")
    print(f"  Valore atteso finale: {formatta_euro(int(np.mean(valori_rimasti) if valori_rimasti else valore_pacco))}")
    print(f"{BLU}{'═'*56}{RESET}\n")

    if input_si_no("  Vuoi giocare ancora? (S/N): "):
        gioca()

# ── Avvio ───────────────────────────────────────────────────────
if __name__ == "__main__":
    gioca()
