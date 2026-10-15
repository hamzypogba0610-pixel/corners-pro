"""
Statistiques descriptives par equipe, par mi-temps, puis sur MC et GLOBAL.
Reference : MULTICORNER 360 v2, section 3.
"""
import numpy as np


def stats_brutes(valeurs):
    """Moyenne, mediane, variance, ecart-type, min, max, quantiles."""
    v = np.array(valeurs, dtype=float)
    if v.size == 0:
        raise ValueError("Liste vide.")
    return {
        "n": int(v.size),
        "moyenne": float(np.mean(v)),
        "mediane": float(np.median(v)),
        "variance": float(np.var(v, ddof=0)),
        "ecart_type": float(np.std(v, ddof=0)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "q10": float(np.quantile(v, 0.10)),
        "q25": float(np.quantile(v, 0.25)),
        "q75": float(np.quantile(v, 0.75)),
        "q90": float(np.quantile(v, 0.90)),
    }


def frequence_zeros(valeurs):
    """Proportion de zeros observes."""
    v = np.array(valeurs, dtype=float)
    if v.size == 0:
        raise ValueError("Liste vide.")
    return float(np.mean(v == 0))


def surdispersion(valeurs):
    """
    Rapport variance / moyenne.
    > 1 : surdispersion (justifie la binomiale negative).
    ~ 1 : compatible Poisson.
    """
    v = np.array(valeurs, dtype=float)
    m = float(np.mean(v))
    if m <= 0:
        return float("inf")
    return float(np.var(v, ddof=0) / m)


def zero_attendu_poisson(valeurs):
    """Sous Poisson de meme moyenne : P(X=0) = exp(-lambda)."""
    v = np.array(valeurs, dtype=float)
    return float(np.exp(-float(np.mean(v))))


def correlation_pearson(x, y):
    """Correlation de Pearson entre deux listes."""
    a = np.array(x, dtype=float)
    b = np.array(y, dtype=float)
    if a.size != b.size or a.size < 2:
        return None
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def correlation_spearman(x, y):
    """Correlation de Spearman (sur les rangs)."""
    a = np.array(x, dtype=float)
    b = np.array(y, dtype=float)
    if a.size != b.size or a.size < 2:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def calcul_mc(hist):
    """Prend une liste de matchs et retourne les series A1, A2, MC."""
    a1 = [m["corners_1h"] for m in hist]
    a2 = [m["corners_2h"] for m in hist]
    mc = [m["corners_1h"] * m["corners_2h"] for m in hist]
    return {"A1": a1, "A2": a2, "MC": mc}


def rapport_equipe(hist, label=""):
    """Rapport descriptif complet pour une equipe."""
    series = calcul_mc(hist)
    rapport = {"label": label}
    for nom, valeurs in series.items():
        rapport[nom] = {
            "stats": stats_brutes(valeurs),
            "freq_zero": frequence_zeros(valeurs),
            "zero_attendu_poisson": zero_attendu_poisson(valeurs),
            "surdispersion": surdispersion(valeurs),
        }
    rapport["corr_A1_A2_pearson"] = correlation_pearson(
        series["A1"], series["A2"]
    )
    rapport["corr_A1_A2_spearman"] = correlation_spearman(
        series["A1"], series["A2"]
    )
    return rapport


def rapport_global(hist_home, hist_away):
    """Combine MC1 (domicile) et MC2 (exterieur) -> GLOBAL = MC1 + MC2."""
    mc1 = [m["corners_1h"] * m["corners_2h"] for m in hist_home]
    mc2 = [m["corners_1h"] * m["corners_2h"] for m in hist_away]
    glob = [a + b for a, b in zip(mc1, mc2)]
    return {
        "MC1": {
            "stats": stats_brutes(mc1),
            "freq_zero": frequence_zeros(mc1),
            "surdispersion": surdispersion(mc1),
        },
        "MC2": {
            "stats": stats_brutes(mc2),
            "freq_zero": frequence_zeros(mc2),
            "surdispersion": surdispersion(mc2),
        },
        "GLOBAL": {
            "stats": stats_brutes(glob),
            "freq_zero": frequence_zeros(glob),
            "surdispersion": surdispersion(glob),
        },
        "corr_MC1_MC2_pearson": correlation_pearson(mc1, mc2),
        "corr_MC1_MC2_spearman": correlation_spearman(mc1, mc2),
  }
