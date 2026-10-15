"""
Modele hierarchique gamma-Poisson avec ponderation temporelle.
Les matchs recents pesent plus lourd (demi-vie en nombre de matchs).
Reference : MULTICORNER 360 v2, section 4.
"""
import numpy as np
from scipy.stats import nbinom
from core.product_law import product_pmf, normalize


KMAX_MIN = 20
KMAX_MARGE = 5
DEMI_VIE_DEFAUT = 8  # en nombre de matchs


def poids_temporels(hist, demi_vie=DEMI_VIE_DEFAUT):
    """
    Calcule un poids pour chaque match, base sur son age.
    Le match le plus RECENT a le poids le plus eleve (proche de 1).
    Formule : poids = 2^(-age / demi_vie).
    Le tri se fait par date (ordre croissant = ancien -> recent).
    Retourne un array de poids dans l'ordre ORIGINAL de hist.
    """
    n = len(hist)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])

    # Trier par date pour trouver l'age relatif
    indices = sorted(range(n), key=lambda i: hist[i]["date"])
    # indices[0] = plus ancien, indices[-1] = plus recent
    # age = position depuis le plus recent (0 pour le plus recent)
    age_par_idx = {}
    for pos, idx in enumerate(indices):
        age_par_idx[idx] = (n - 1) - pos  # 0 pour le plus recent

    poids = np.array([
        2.0 ** (-age_par_idx[i] / float(demi_vie))
        for i in range(n)
    ])
    return poids


def estimate_alpha_beta(valeurs, poids=None):
    """
    Methode des moments PONDEREE sur la NB marginale :
      mu = alpha / beta
      var = mu + mu^2 / alpha
    => alpha = mu^2 / (var - mu), beta = mu / (var - mu)
    """
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
        # Sous-dispersion : on impose une dispersion minimale
        var = mu * 1.5
    alpha = mu * mu / (var - mu)
    beta = mu / (var - mu)
    return float(max(alpha, 1e-6)), float(max(beta, 1e-9))


def posterior_alpha_beta(valeurs, alpha0, beta0, poids=None):
    """
    Posterior conjugue gamma-Poisson PONDERE.
    Prior : lambda ~ Gamma(alpha0, beta0)
    Observations ponderees : x_1..x_n de poids w_i
    Posterior : Gamma(alpha0 + sum(w_i * x_i), beta0 + sum(w_i))
    """
    v = np.array(valeurs, dtype=float)
    if poids is None:
        w = np.ones_like(v)
    else:
        w = np.asarray(poids, dtype=float)
    s_p = float(np.sum(w * v))
    s_w = float(np.sum(w))
    return float(alpha0 + s_p), float(beta0 + s_w)


def nb_pmf(k_max, alpha, beta):
    """Loi binomiale negative predictive, normalisee."""
    p = beta / (beta + 1.0)
    ks = np.arange(k_max + 1)
    pmf = nbinom.pmf(ks, alpha, p)
    s = pmf.sum()
    if s <= 0:
        raise ValueError("PMF NB nulle, parametres invalides.")
    return pmf / s


def _km(hist, cle):
    """max observe + marge, borne basse KMAX_MIN."""
    vals = [m[cle] for m in hist]
    return max(max(vals) + KMAX_MARGE, KMAX_MIN)


def build_pmf_equipe(hist, demi_vie=DEMI_VIE_DEFAUT):
    """
    Construit les PMF de A1 et A2 avec shrinkage et ponderation temporelle.
    Retourne dict {A1: pmf, A2: pmf, alpha_beta: {...}, poids: array}.
    """
    a1 = [m["corners_1h"] for m in hist]
    a2 = [m["corners_2h"] for m in hist]

    w = poids_temporels(hist, demi_vie=demi_vie)

    # Ligue locale : on pondere aussi par les poids temporels
    ligue = a1 + a2
    poids_ligue = np.concatenate([w, w])
    alpha0, beta0 = estimate_alpha_beta(ligue, poids=poids_ligue)

    # Posterior par equipe, pondere
    aa1, bb1 = posterior_alpha_beta(a1, alpha0, beta0, poids=w)
    aa2, bb2 = posterior_alpha_beta(a2, alpha0, beta0, poids=w)

    km1 = _km(hist, "corners_1h")
    km2 = _km(hist, "corners_2h")

    pmf_a1 = nb_pmf(km1, aa1, bb1)
    pmf_a2 = nb_pmf(km2, aa2, bb2)

    return {
        "A1": pmf_a1,
        "A2": pmf_a2,
        "poids": w,
        "demi_vie": demi_vie,
        "alpha_beta": {
            "prior": (alpha0, beta0),
            "posterior_A1": (aa1, bb1),
            "posterior_A2": (aa2, bb2),
        },
    }


def build_joint(pmf_a1, pmf_a2):
    """Loi jointe P(X, Y) sous independance (test de copule plus tard)."""
    joint = np.outer(pmf_a1, pmf_a2)
    return normalize(joint)


def build_mc_global(hist_home, hist_away, demi_vie=DEMI_VIE_DEFAUT):
    """
    Pipeline complet pour un match.
    """
    from core.product_law import convolve

    home = build_pmf_equipe(hist_home, demi_vie=demi_vie)
    away = build_pmf_equipe(hist_away, demi_vie=demi_vie)

    joint_home = build_joint(home["A1"], home["A2"])
    joint_away = build_joint(away["A1"], away["A2"])

    mc1_pmf = product_pmf(joint_home)
    mc2_pmf = product_pmf(joint_away)
    global_pmf = convolve(mc1_pmf, mc2_pmf)

    return {
        "home": home,
        "away": away,
        "joint_home": joint_home,
        "joint_away": joint_away,
        "mc1_pmf": mc1_pmf,
        "mc2_pmf": mc2_pmf,
        "global_pmf": global_pmf,
    }
