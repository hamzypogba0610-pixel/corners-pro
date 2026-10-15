"""
Loi exacte du produit M = X·Y et de GLOBAL = MC1 + MC2.
Aucune simulation : sommes sur tous les couples (x, y) tels que x*y = m.
Reference : MULTICORNER 360 v2, section 4.
"""
import numpy as np


def normalize(joint):
    s = joint.sum()
    if s <= 0:
        raise ValueError("Loi jointe vide.")
    return joint / s


def product_pmf(joint):
    """Loi exacte de M = X * Y a partir de la loi jointe P(X=x, Y=y)."""
    joint = normalize(joint)
    xmax, ymax = joint.shape[0] - 1, joint.shape[1] - 1
    pmf = {}
    for x in range(xmax + 1):
        for y in range(ymax + 1):
            m = x * y
            pmf[m] = pmf.get(m, 0.0) + float(joint[x, y])
    return pmf


def convolve(pmf_a, pmf_b):
    """Convolution discrete : loi de GLOBAL = MC1 + MC2 (independance)."""
    out = {}
    for a, pa in pmf_a.items():
        for b, pb in pmf_b.items():
            s = a + b
            out[s] = out.get(s, 0.0) + pa * pb
    return out


def moments(pmf):
    """Moyenne, variance, ecart-type, mediane, quantiles, min, max."""
    if not pmf:
        raise ValueError("pmf vide.")
    keys = sorted(pmf.keys())
    vals = np.array([pmf[k] for k in keys], dtype=float)
    vals = vals / vals.sum()

    mean = float(np.sum(np.array(keys) * vals))
    var = float(np.sum((np.array(keys) - mean) ** 2 * vals))

    cum = np.cumsum(vals)
    def q(p):
        idx = np.searchsorted(cum, p)
        idx = min(idx, len(keys) - 1)
        return int(keys[idx])

    return {
        "mean": mean,
        "var": var,
        "std": float(np.sqrt(var)),
        "min": int(keys[0]),
        "max": int(keys[-1]),
        "q10": q(0.10), "q25": q(0.25), "median": q(0.50),
        "q75": q(0.75), "q90": q(0.90),
    }


def p_zero(pmf):
    """P(M = 0). Regle du zero : X=0 ou Y=0 annule le produit."""
    return float(pmf.get(0, 0.0))


def p_over(pmf, line):
    """Over : resultat STRICTEMENT superieur a la ligne."""
    if line == int(line):
        raise ValueError("Ligne entiere : gerer explicitement le push.")
    return float(sum(p for m, p in pmf.items() if m > line))


def p_under(pmf, line):
    """Under d'une ligne reelle (avec .5) : 1 - P(Over)."""
    return 1.0 - p_over(pmf, line)


def variance_over_mean(joint, axis):
    """Rapport variance/moyenne (indice de surdispersion)."""
    joint = normalize(joint)
    marg = joint.sum(axis=axis)
    ks = np.arange(len(marg))
    mean = float(np.sum(ks * marg))
    var = float(np.sum((ks - mean) ** 2 * marg))
    return var / mean if mean > 0 else float("inf")


def margin_removal_proportional(over_price, under_price):
    """Retrait de marge par normalisation proportionnelle."""
    p_over_imp = 1.0 / over_price
    p_under_imp = 1.0 / under_price
    total = p_over_imp + p_under_imp
    return (p_over_imp / total, p_under_imp / total, total - 1.0)


def margin_removal_power(over_price, under_price, tol=1e-8, max_iter=200):
    """
    Retrait de marge par methode puissance (bissection sur k).
    f(k) = (1/over)^k + (1/under)^k - 1 est DECROISSANTE en k.
    Donc : si f > 0 -> k trop petit -> on remonte lo. Sinon on baisse hi.
    """
    a, b = 1.0 / over_price, 1.0 / under_price
    lo, hi = 0.5, 3.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        f = a ** mid + b ** mid - 1.0
        if abs(f) < tol:
            break
        if f > 0:
            lo = mid
        else:
            hi = mid
    k = (lo + hi) / 2
    p_over = a ** k
    p_under = b ** k
    return (p_over, p_under, 1.0 - (p_over + p_under))
