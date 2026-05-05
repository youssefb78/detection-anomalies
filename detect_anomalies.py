import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy import stats
import matplotlib.pyplot as plt
import json
import csv
from datetime import datetime

# ── 1. PARSING DES SPÉCIFICATIONS ─────────────────────────
print("Chargement des spécifications...")
with open("specs.json", "r", encoding="utf-8") as f:
    specs = json.load(f)

exigences = specs["exigences"]
print(f"✅ {len(exigences)} exigences chargées depuis specs.json")
for exg in exigences:
    print(f"   → {exg['id']} : {exg['description']}")

# ── 2. GÉNÉRATION DES DONNÉES ──────────────────────────────
np.random.seed(42)
n = 200
temps = pd.date_range(start="2025-01-01", periods=n, freq="h")
temperature = np.random.normal(loc=80, scale=5, size=n)

# Injection d'anomalies
temperature[30] = 150
temperature[80] = 20
temperature[150] = 145

df = pd.DataFrame({"timestamp": temps, "temperature": temperature})

# ── 3. GÉNÉRATION AUTOMATIQUE DES CAMPAGNES DE TESTS ──────
print("\nGénération des campagnes de tests...")
campagne = []
for exg in exigences:
    for i, row in df.iterrows():
        cas = {
            "exigence_id": exg["id"],
            "timestamp": str(row["timestamp"]),
            "valeur": round(row["temperature"], 4),
            "seuil_max": exg["seuil_max"],
            "seuil_min": exg["seuil_min"],
        }
        # Évaluation du critère
        if exg["seuil_max"] is not None and row["temperature"] > exg["seuil_max"]:
            cas["resultat"] = "ECHEC"
        elif exg["seuil_min"] is not None and row["temperature"] < exg["seuil_min"]:
            cas["resultat"] = "ECHEC"
        else:
            cas["resultat"] = "PASS"
        campagne.append(cas)

nb_tests = len(campagne)
nb_echecs = sum(1 for c in campagne if c["resultat"] == "ECHEC")
print(f"✅ {nb_tests} cas de tests générés automatiquement")
print(f"   → PASS : {nb_tests - nb_echecs} | ECHEC : {nb_echecs}")

# ── 4. DÉTECTION Z-SCORE ───────────────────────────────────
df["z_score"] = np.abs(stats.zscore(df["temperature"]))
df["anomalie_zscore"] = df["z_score"] > 3

# ── 5. DÉTECTION ISOLATION FOREST ─────────────────────────
model = IsolationForest(contamination=0.05, random_state=42)
df["anomalie_if"] = model.fit_predict(df[["temperature"]]) == -1

# ── 6. COMBINAISON ─────────────────────────────────────────
df["anomalie_finale"] = df["anomalie_zscore"] | df["anomalie_if"]

# ── 7. TRAÇABILITÉ EXIGENCES → TESTS → RÉSULTATS ──────────
print("\nGénération du rapport de traçabilité...")

rapport_tracabilite = []
for exg in exigences:
    tests_exg = [c for c in campagne if c["exigence_id"] == exg["id"]]
    nb_pass = sum(1 for t in tests_exg if t["resultat"] == "PASS")
    nb_echec = sum(1 for t in tests_exg if t["resultat"] == "ECHEC")
    rapport_tracabilite.append({
        "exigence_id": exg["id"],
        "description": exg["description"],
        "nb_tests": len(tests_exg),
        "nb_pass": nb_pass,
        "nb_echec": nb_echec,
        "taux_conformite": f"{nb_pass/len(tests_exg)*100:.1f}%",
        "statut": "CONFORME" if nb_echec == 0 else "NON CONFORME"
    })

# ── 8. RAPPORT FORMEL ──────────────────────────────────────
nb_anomalies = df["anomalie_finale"].sum()
print("\n" + "=" * 60)
print("       RAPPORT DE VALIDATION FORMEL")
print("=" * 60)
print(f"Date génération     : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"Total mesures       : {len(df)}")
print(f"Anomalies détectées : {nb_anomalies}")
print(f"Taux anomalies      : {nb_anomalies/len(df)*100:.1f}%")
print("=" * 60)
print("\nTRAÇABILITÉ EXIGENCES → TESTS → RÉSULTATS :")
print("-" * 60)
for r in rapport_tracabilite:
    print(f"{r['exigence_id']} | {r['nb_tests']} tests | "
          f"PASS:{r['nb_pass']} ECHEC:{r['nb_echec']} | "
          f"Conformité:{r['taux_conformite']} | {r['statut']}")
print("=" * 60)

# Sauvegarde CSV tracabilité
with open("rapport_tracabilite.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rapport_tracabilite[0].keys())
    writer.writeheader()
    writer.writerows(rapport_tracabilite)

# Sauvegarde campagne de tests
with open("campagne_tests.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=campagne[0].keys())
    writer.writeheader()
    writer.writerows(campagne)

# ── 9. GRAPHIQUE ───────────────────────────────────────────
plt.figure(figsize=(14, 5))
plt.plot(df["timestamp"], df["temperature"],
         label="Température", color="steelblue", linewidth=1.5)
plt.scatter(
    df[df["anomalie_finale"]]["timestamp"],
    df[df["anomalie_finale"]]["temperature"],
    color="red", zorder=5, label="Anomalie détectée", s=80
)
plt.axhline(y=120, color="orange", linestyle="--",
            alpha=0.7, label="Seuil max EXG-001 (120°C)")
plt.axhline(y=30, color="purple", linestyle="--",
            alpha=0.7, label="Seuil min EXG-002 (30°C)")
plt.axhline(y=80, color="green", linestyle="--",
            alpha=0.5, label="Moyenne normale")
plt.title("Rapport de Validation — Détection d'Anomalies IVVQ")
plt.xlabel("Temps")
plt.ylabel("Température (°C)")
plt.legend()
plt.tight_layout()
plt.savefig("rapport_anomalies.png")
plt.show()

print("\n✅ Fichiers générés :")
print("   → rapport_anomalies.png")
print("   → rapport_tracabilite.csv")
print("   → campagne_tests.csv")