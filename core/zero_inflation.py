"""
Test et modelisation de l'inflation de zeros.
Quand P(X=0) observe depasse P(X=0) attendu sous NB, on bascule sur
un modele zero-inflated negative binomial (ZINB).
Reference : MULTICORNER 360 v2, section 4, point 6.
"""
import numpy as np
from scipy.stats import nbinom


SEUIL_ECART = 0.05       # 5 points de proba d'ecart minimum
SHRINKAGE_K = 5.0        # plus n est petit, plus on freine pi
PI_MAX = 0.4             # borne haute pour pi
PI_SEUIL_ACTIF = 0.01    # sous ce seuil, on considere l'effet nul


def _p_zero_nb(alpha, beta):
    """P(X=0) sous NB(alpha, beta)."""
    return float(nbinom.pmf(0, alpha, beta / (beta + 1.0)))


def tester_inflation(valeurs, alpha, beta):
    """
    Compare P(X=0) observe et P(X=0) attendu sous NB.
    Retourne dict avec l'ecart, la valeur de pi (shrinkee), et le statut.
    """
    v = np.array(valeurs, dtype=float)
    n = len(v)
    if n == 0:
        return {
            "n": 0, "p_zero_obs": 0.0, "p_zero_nb": 0.0,
            "ecart": 0.0, "pi": 0.0, "pi_brut": 0.0,
            "inflation": False, "raison": "aucune donnee",
        }
    p_obs = float(np.mean(v == 0))
    p_nb = _p_zero_nb(alpha, beta)
    ecart = p_obs - p_nb
    if ecart < SEUIL_ECART:
        return {
            "n": n, "p_zero_obs": p_obs, "p_zero_nb": p_nb,
            "ecart": ecart, "pi": 0.0, "pi_brut": 0.0,
            "inflation": False, "raison": "ecart sous le seuil",
        }
    denom = 1.0 - p_nb
    if denom <= 1e-6:
        pi_brut = 0.0
    else:
        pi_brut = (p_obs - p_nb) / denom
    pi_brut = max(0.0, min(PI_MAX, pi_brut))
    # Shrinkage : avec n petit, on reduit pi
    w = n / (n + SHRINKAGE_K)
    pi = pi_brut * w
    return {
        "n": n, "p_zero_obs": p_obs, "p_zero_nb": p_nb,
        "ecart": ecart, "pi": pi, "pi_brut": pi_brut,
        "inflation": pi > PI_SEUIL_ACTIF,
        "raison": (
            "inflation detectee" if pi > PI_SEUIL_ACTIF
            else "shrinkage annule l'effet"
        ),
    }


def pmf_zinb(k_max, alpha, beta, pi):
    """
    Loi zero-inflated NB.
    P(X=0) = pi + (1-pi)*NB(0)
    P(X=k>0) = (1-pi)*NB(k)
    """
    p = beta / (beta + 1.0)
    ks = np.arange(k_max + 1)
    pmf_nb = nbinom.pmf(ks, alpha, p)
    s = pmf_nb.sum()
    if s <= 0:
        raise ValueError("PMF NB nulle, parametres invalides.")
    pmf_nb = pmf_nb / s
    if pi <= 0:
        return pmf_nb
    pmf = (1.0 - pi) * pmf_nb
    pmf[0] += pi
    total = pmf.sum()
    if total <= 0:
        raise ValueError("PMF ZINB vide.")
    return pmf / total
