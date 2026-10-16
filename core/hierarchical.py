"""
Modele hierarchique gamma-Poisson avec ponderation temporelle,
effet score 2e MT, inflation de zeros, poids reduit pour les matchs
globaux ('N'), et facteur de ligue.
"""
import numpy as np
from scipy.stats import nbinom
from core.product_law import product_pmf, normalize
from core.zero_inflation import tester_inflation, pmf_zinb


KMAX_MIN = 20
KMAX_MARGE = 5
DEMI_VIE_DEFAUT = 8
FACTEUR_LIEU_GLOBAL = 0.6


def poids_temporels(hist, demi_vie=DEMI_VIE_DEFAUT):
    n = len(hist)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])
    indices = sorted(range(n), key=lambda i: hist[i]["date"])
    age_par_idx = {}
    for pos, idx in enumerate(indices):
        age_par_idx[idx] = (n - 1) - pos
    poids_temps = np.array([
        2.0 ** (-age_par_idx[i] / float(demi_vie))
        for i in range(n)
    ])
    facteur_lieu = np.array([
        FACTEUR_LIEU_GLOBAL if hist[i].get("venue") == "N" else 1.0
        for i in range(n)
    ])
    return poids_temps * facteur_lieu


def estimate_alpha_beta(valeurs, poids=None):
    v = np.array(valeurs, dtype=float)
    if v.size == 0:
        raise ValueError("Liste vide.")
    if poids is None:
        w = np.ones_like(v)
    else:
        w = np.asarray(poids, dtype=float)
        if len(w) != len(v):
            raise ValueError("poids et valeurs de tailles differentes")
    sw = float(np.sum(w))
    if sw <= 0:
        raise ValueError("Somme des poids nulle.")
    mu = float(np.sum(w * v) / sw)
    var = float(np.sum(w * (v - mu) ** 2) / sw)
    if mu <= 0:
        return 1e-6, 1.0
    if var <= mu * 1.0001:
        var = mu * 1.5
    alpha = mu * mu / (var - mu)
    beta = mu / (var - mu)
    return float(max(alpha, 1e-6)), float(max(beta, 1e-9))


def posterior_alpha_beta(valeurs, alpha0, beta0, poids=None):
    v = np.array(valeurs, dtype=float)
    if poids is None:
        w = np.ones_like(v)
    else:
        w = np.asarray(poids, dtype=float)
    s_p = float(np.sum(w * v))
    s_w = float(np.sum(w))
    return float(alpha0 + s_p), float(beta0 + s_w)


def nb_pmf(k_max, alpha, beta):
    p = beta / (beta + 1.0)
    ks = np.arange(k_max + 1)
    pmf = nbinom.pmf(ks, alpha, p)
    s = pmf.sum()
    if s <= 0:
        raise ValueError("PMF NB nulle, parametres invalides.")
    return pmf / s


def _km(hist, cle):
    vals = [m[cle] for m in hist]
    return max(max(vals) + KMAX_MARGE, KMAX_MIN)


def build_pmf_equipe(hist, demi_vie=DEMI_VIE_DEFAUT, facteur_2h=1.0,
                     facteur_ligue=1.0, utiliser_zinb=True):
    """
    PMF de A1 et A2 avec shrinkage, ponderation temporelle,
    effet score, facteur de ligue, et inflation de zeros.
    """
    a1 = [m["corners_1h"] for m in hist]
    a2 = [m["corners_2h"] for m in hist]

    w = poids_temporels(hist, demi_vie=demi_vie)

    ligue = a1 + a2
    poids_ligue = np.concatenate([w, w])
    alpha0, beta0 = estimate_alpha_beta(ligue, poids=poids_ligue)

    aa1, bb1 = posterior_alpha_beta(a1, alpha0, beta0, poids=w)
    aa2, bb2 = posterior_alpha_beta(a2, alpha0, beta0, poids=w)
    # Facteur de ligue s'applique aux DEUX mi-temps.
    # Facteur effet score 2e MT s'applique a la 2e MT uniquement.
    f_ligue = float(facteur_ligue)
    aa1_ajuste = aa1 * f_ligue
    aa2_ajuste = aa2 * float(facteur_2h) * f_ligue

    km1 = _km(hist, "corners_1h")
    km2 = _km(hist, "corners_2h")

    info_z1 = {"inflation": False, "pi": 0.0, "raison": "desactive"}
    info_z2 = {"inflation": False, "pi": 0.0, "raison": "desactive"}
    if utiliser_zinb:
        info_z1 = tester_inflation(a1, aa1_ajuste, bb1)
        info_z2 = tester_inflation(a2, aa2_ajuste, bb2)
        if info_z1["inflation"]:
            pmf_a1 = pmf_zinb(km1, aa1_ajuste, bb1, info_z1["pi"])
        else:
            pmf_a1 = nb_pmf(km1, aa1_ajuste, bb1)
        if info_z2["inflation"]:
            pmf_a2 = pmf_zinb(km2, aa2_ajuste, bb2, info_z2["pi"])
        else:
            pmf_a2 = nb_pmf(km2, aa2_ajuste, bb2)
    else:
        pmf_a1 = nb_pmf(km1, aa1_ajuste, bb1)
        pmf_a2 = nb_pmf(km2, aa2_ajuste, bb2)

    return {
        "A1": pmf_a1,
        "A2": pmf_a2,
        "poids": w,
        "demi_vie": demi_vie,
        "facteur_2h": float(facteur_2h),
        "facteur_ligue": f_ligue,
        "zero_inflation": {"A1": info_z1, "A2": info_z2},
        "alpha_beta": {
            "prior": (alpha0, beta0),
            "posterior_A1": (aa1, bb1),
            "posterior_A2": (aa2, bb2),
            "posterior_A1_ajuste": (aa1_ajuste, bb1),
            "posterior_A2_ajuste": (aa2_ajuste, bb2),
        },
    }


def build_joint(pmf_a1, pmf_a2):
    joint = np.outer(pmf_a1, pmf_a2)
    return normalize(joint)


def build_mc_global(hist_home, hist_away, demi_vie=DEMI_VIE_DEFAUT,
                    facteur_home_2h=1.0, facteur_away_2h=1.0,
                    facteur_ligue=1.0, utiliser_zinb=True):
    from core.product_law import convolve
    home = build_pmf_equipe(hist_home, demi_vie=demi_vie,
                            facteur_2h=facteur_home_2h,
                            facteur_ligue=facteur_ligue,
                            utiliser_zinb=utiliser_zinb)
    away = build_pmf_equipe(hist_away, demi_vie=demi_vie,
                            facteur_2h=facteur_away_2h,
                            facteur_ligue=facteur_ligue,
                            utiliser_zinb=utiliser_zinb)
    joint_home = build_joint(home["A1"], home["A2"])
    joint_away = build_joint(away["A1"], away["A2"])
    mc1_pmf = product_pmf(joint_home)
    mc2_pmf = product_pmf(joint_away)
    global_pmf = convolve(mc1_pmf, mc2_pmf)
    return {
        "home": home, "away": away,
        "joint_home": joint_home, "joint_away": joint_away,
        "mc1_pmf": mc1_pmf, "mc2_pmf": mc2_pmf, "global_pmf": global_pmf,
                    }
