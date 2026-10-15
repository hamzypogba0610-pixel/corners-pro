"""
Modele hierarchique gamma-Poisson (binomiale negative conjuguee).
Estime les intensites par equipe et par mi-temps avec shrinkage
vers la moyenne de ligue. Genere la loi jointe P(X=x, Y=y).
Reference : MULTICORNER 360 v2, section 4.
"""
import numpy as np
from scipy.stats import nbinom
from core.product_law import product_pmf, normalize


KMAX_MIN = 20
KMAX_MARGE = 5


def estimate_alpha_beta(valeurs):
    """
    Methode des moments sur la NB marginale :
      mu = alpha / beta
      var = mu + mu^2 / alpha
    => alpha = mu^2 / (var - mu), beta = mu / var_shrunk... (voir code)
    Retourne (alpha, beta). Si var <= mu, on renvoie une borne prudente.
    """
    v = np.array(valeurs, dtype=float)
    if v.size == 0:
        raise ValueError("Liste vide.")
    mu = float(np.mean(v))
    var = float(np.var(v, ddof=0))
    if mu <= 0:
        return 1e-6, 1.0
    if var <= mu * 1.0001:
        # Sous-dispersion : on force une dispersion minimale pour eviter
        # un alpha infini (ce qui reviendrait a Poisson pur).
        var = mu * 1.5
    alpha = mu * mu / (var - mu)
    beta = mu / (var - mu) * mu  # beta = mu / (var - mu) * mu -> coherent
    beta = mu / (var - mu) * 1.0
    beta = 1.0 / (var - mu) * mu  # simplification : beta = mu / (var - mu) ? Non.
    # Reprenons proprement :
    # mu = alpha/beta ; var = mu + mu^2/alpha
    # => alpha = mu^2 / (var - mu)
    # => beta = alpha / mu = mu / (var - mu)
    alpha = mu * mu / (var - mu)
    beta = mu / (var - mu)
    return float(max(alpha, 1e-6)), float(max(beta, 1e-9))


def posterior_alpha_beta(valeurs, alpha0, beta0):
    """
    Posterior conjugue gamma-Poisson.
    Prior : lambda ~ Gamma(alpha0, beta0)
    Observations : x_1..x_n ~ Poisson(lambda)
    Posterior : Gamma(alpha0 + sum x, beta0 + n)
    """
    v = np.array(valeurs, dtype=float)
    n = v.size
    s = float(np.sum(v))
    return float(alpha0 + s), float(beta0 + n)


def nb_pmf(k_max, alpha, beta):
    """
    Loi binomiale negative predictive (marginale apres integration du lambda).
    P(X = k) = C(k + alpha - 1, k) * (beta/(beta+1))^alpha * (1/(beta+1))^k
    Formule scipy : nbinom.pmf(k, n=alpha, p=beta/(beta+1))
    Retourne un vecteur np de taille k_max+1, normalise.
    """
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


def build_pmf_equipe(hist):
    """
    Construit les PMF de A1 (1re MT) et A2 (2e MT) avec shrinkage.
    Retourne un dict {A1: pmf, A2: pmf, alpha_beta: {...}}.
    """
    a1 = [m["corners_1h"] for m in hist]
    a2 = [m["corners_2h"] for m in hist]

    # Ligue locale = union des deux series (pour estimer un prior commun)
    ligue = a1 + a2
    alpha0, beta0 = estimate_alpha_beta(ligue)

    # Posterior par equipe sur ses propres observations (ici = hist)
    # On traite chaque mi-temps separement mais avec le meme prior
    aa1, bb1 = posterior_alpha_beta(a1, alpha0, beta0)
    aa2, bb2 = posterior_alpha_beta(a2, alpha0, beta0)

    km1 = _km(hist, "corners_1h")
    km2 = _km(hist, "corners_2h")

    pmf_a1 = nb_pmf(km1, aa1, bb1)
    pmf_a2 = nb_pmf(km2, aa2, bb2)

    return {
        "A1": pmf_a1,
        "A2": pmf_a2,
        "alpha_beta": {
            "prior": (alpha0, beta0),
            "posterior_A1": (aa1, bb1),
            "posterior_A2": (aa2, bb2),
        },
    }


def build_joint(pmf_a1, pmf_a2):
    """
    Loi jointe P(X, Y) sous independance (a tester plus tard par copule).
    Matrice (len(pmf_a1), len(pmf_a2)).
    """
    k1 = len(pmf_a1)
    k2 = len(pmf_a2)
    # On tronque/pad a une taille commune pour la matrice
    kmax1 = k1 - 1
    kmax2 = k2 - 1
    joint = np.outer(pmf_a1, pmf_a2)
    return normalize(joint)


def build_mc_global(hist_home, hist_away):
    """
    Pipeline complet pour un match :
    1. PMF A1, A2 pour equipe domicile -> joint -> loi de MC1 = A1*A2
    2. Idem equipe exterieur -> MC2
    3. Convolution -> GLOBAL (sous independance MC1/MC2)
    Retourne dict avec mc1_pmf, mc2_pmf, global_pmf, et les PMF intermediaires.
    """
    from core.product_law import convolve

    home = build_pmf_equipe(hist_home)
    away = build_pmf_equipe(hist_away)

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
