import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy import stats
import matplotlib.pyplot as plt

# ── 1. GÉNÉRATION DES DONNÉES ──────────────────────────────
np.random.seed(42)
n = 200

temps = pd.date_range(start="2025-01-01", periods=n, freq="h")
temperature = np.random.normal(loc=80, scale=5, size=n)

# Injection d'anomalies réalistes
temperature[30] = 150   # pic anormal
temperature[80] = 20    # chute anormale
temperature[150] = 145  # autre pic

df = pd.DataFrame({"timestamp": temps, "temperature": temperature})

# ── 2. DÉTECTION Z-SCORE ───────────────────────────────────
df["z_score"] = np.abs(stats.zscore(df["temperature"]))
df["anomalie_zscore"] = df["z_score"] > 3  # seuil = 3 écarts-types

# ── 3. DÉTECTION ISOLATION FOREST ─────────────────────────
model = IsolationForest(contamination=0.05, random_state=42)
df["anomalie_if"] = model.fit_predict(df[["temperature"]]) == -1

# ── 4. COMBINAISON DES DEUX MÉTHODES ──────────────────────
df["anomalie_finale"] = df["anomalie_zscore"] | df["anomalie_if"]

# ── 5. RAPPORT ────────────────────────────────────────────
nb_anomalies = df["anomalie_finale"].sum()
print("=" * 50)
print("       RAPPORT DE DÉTECTION D'ANOMALIES")
print("=" * 50)
print(f"Total mesures analysées  : {len(df)}")
print(f"Anomalies détectées      : {nb_anomalies}")
print(f"Taux d'anomalies         : {nb_anomalies/len(df)*100:.1f}%")
print("=" * 50)
print("\nDétail des anomalies :")
print(df[df["anomalie_finale"]][["timestamp","temperature","z_score"]].to_string(index=False))

# ── 6. GRAPHIQUE ───────────────────────────────────────────
plt.figure(figsize=(14, 5))
plt.plot(df["timestamp"], df["temperature"], label="Température", color="steelblue", linewidth=1.5)
plt.scatter(
    df[df["anomalie_finale"]]["timestamp"],
    df[df["anomalie_finale"]]["temperature"],
    color="red", zorder=5, label="Anomalie détectée", s=80
)
plt.axhline(y=80, color="green", linestyle="--", alpha=0.5, label="Moyenne normale")
plt.title("Détection d'Anomalies — Température Moteur")
plt.xlabel("Temps")
plt.ylabel("Température (°C)")
plt.legend()
plt.tight_layout()
plt.savefig("rapport_anomalies.png")
plt.show()
print("\nGraphique sauvegardé : rapport_anomalies.png")