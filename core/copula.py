"""
Copule de Frank pour la dependance entre mi-temps (A1/A2)
et entre equipes (MC1/MC2).
Reference : MULTICORNER 360 v2, section 4, point 4.
"""
import numpy as np
from scipy.integrate import quad
from scipy.stats import kendalltau
from core.product_law import normalize


THETA_MIN = -30.0
THETA_MAX = 30.0
TAU_SEUIL = 0.05


def _C_frank(u, v, theta):
    if u <= 0.0 or v <= 0.0:
        return 0.0
    if u >= 1.0:
        return float(v)
    if v >= 1.0:
        return float(u)
    if abs(theta) < 1e-8:
        return float(u * v)
    exp_neg_theta = np.exp(-theta)
    num = (np.exp(-theta * u) - 1.0) * (np.exp(-theta * v) - 1.0)
    den = exp_neg_theta - 1.0
    if den == 0:
        return float(u * v)
    arg = 1.0 + num / den
    if arg <= 0:
        return 0.0
    return float(-1.0 / theta * np.log(arg))


def pmf_joint_frank(pmf_x, pmf_y, theta):
    """
    Loi jointe P(X=x, Y=y) via copule de Frank.
    Differences finies sur la fonction copule.
    """
    px = np.asarray(pmf_x, dtype=float)
    py = np.asarray(pmf_y, dtype=float)
    px = px / px.sum()
    py = py / py.sum()

    fx = np.cumsum(px)
    fy = np.cumsum(py)
    fx_prev = np.concatenate([[0.0], fx[:-1]])
    fy_prev = np.concatenate([[0.0], fy[:-1]])

    nx = len(px); ny = len(py)
    joint = np.zeros((nx, ny))
    for i in range(nx):
        u1, u0 = fx[i], fx_prev[i]
        for j in range(ny):
            v1, v0 = fy[j], fy_prev[j]
            joint[i, j] = (
                _C_frank(u1, v1, theta)
                - _C_frank(u0, v1, theta)
                - _C_frank(u1, v0, theta)
                + _C_frank(u0, v0, theta)
            )
    joint = np.clip(joint, 0.0, None)
    return normalize(joint)


def _D1_debye(theta):
    if abs(theta) < 1e-8:
        return 1.0
    f = lambda t: t / (np.exp(t) - 1.0) if abs(t) > 1e-12 else 1.0
    val, _ = quad(f, 0.0, theta, limit=100)
    return float(val / theta)


def tau_from_theta(theta):
    if abs(theta) < 1e-8:
        return 0.0
    return float(1.0 - 4.0 / theta + 4.0 / theta * _D1_debye(theta))


def theta_from_tau(tau):
    if tau <= tau_from_theta(THETA_MIN):
        return THETA_MIN
    if tau >= tau_from_theta(THETA_MAX):
        return THETA_MAX
    lo, hi = THETA_MIN, THETA_MAX
    for _ in range(80):
        mid = (lo + hi) / 2
        t_mid = tau_from_theta(mid)
        if t_mid < tau:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2)


def estimer_theta(x_obs, y_obs):
    x = np.asarray(x_obs, dtype=float)
    y = np.asarray(y_obs, dtype=float)
    if len(x) < 4:
        return 0.0, 0.0, True
    tau, _ = kendalltau(x, y)
    if np.isnan(tau):
        tau = 0.0
    tau = float(tau)
    if abs(tau) < TAU_SEUIL:
        return 0.0, tau, True
    theta = theta_from_tau(tau)
    return theta, tau, False


def joint_avec_copule(pmf_x, pmf_y, x_obs, y_obs):
    """Joint 2D via copule estimee sur les observations."""
    theta, tau, indep_ok = estimer_theta(x_obs, y_obs)
    if indep_ok:
        joint = normalize(np.outer(pmf_x, pmf_y))
    else:
        joint = pmf_joint_frank(pmf_x, pmf_y, theta)
    return joint, {"theta": theta, "tau": tau, "independance_ok": indep_ok}


# ---------------------------------------------------------------------------
# CORRELATION INTER-EQUIPES (proxy via xg_contre)
# ---------------------------------------------------------------------------

RHO_SEUIL = 0.10


def rho_vers_tau_gaussien(rho):
    """
    Conversion rho Pearson -> tau Kendall sous copule gaussienne.
    Formule classique : tau = (2/pi) * arcsin(rho).
    """
    rho = max(-0.999, min(0.999, float(rho)))
    return float(2.0 / np.pi * np.arcsin(rho))


def estimer_correlation_inter_equipes(hist_home, hist_away):
    """
    Estime la correlation inter-equipes.
    Proxy : corr(corners_totaux_de_l_equipe, xg_contre_de_l_equipe).
    Un xg_contre eleve = adversaire offensif = potentiellement
    beaucoup de corners des deux cotes.
    Combine les deux historiques pour plus de donnees.
    Retourne (rho, n, methode).
    """
    c_tot = []
    xg_c = []
    for m in hist_home + hist_away:
        c_tot.append(m["corners_1h"] + m["corners_2h"])
        xg_c.append(m["xg_contre"])
    c_arr = np.array(c_tot, dtype=float)
    x_arr = np.array(xg_c, dtype=float)
    if len(c_arr) < 10:
        return 0.0, len(c_arr), "trop peu de donnees"
    if np.std(c_arr) == 0 or np.std(x_arr) == 0:
        return 0.0, len(c_arr), "variance nulle"
    rho = float(np.corrcoef(c_arr, x_arr)[0, 1])
    if np.isnan(rho):
        return 0.0, len(c_arr), "calcul impossible"
    return rho, len(c_arr), "pearson"


def joint_mc_via_copule(mc1_pmf, mc2_pmf, rho):
    """
    Construit la loi jointe de (MC1, MC2) via une copule de Frank
    calibree sur rho.
    Si |rho| < RHO_SEUIL, retourne l'independance pure.
    Retourne (joint_2d, info).
    """
    if abs(rho) < RHO_SEUIL:
        joint = normalize(np.outer(mc1_pmf, mc2_pmf))
        return joint, {
            "rho": rho, "tau": 0.0, "theta": 0.0,
            "independance_ok": True, "methode": "independance",
        }
    tau = rho_vers_tau_gaussien(rho)
    theta = theta_from_tau(tau)
    joint = pmf_joint_frank(mc1_pmf, mc2_pmf, theta)
    return joint, {
        "rho": rho, "tau": tau, "theta": theta,
        "independance_ok": False, "methode": "copule_frank",
    }


def global_depuis_joint_mc(joint_2d):
    """
    Loi de GLOBAL = MC1 + MC2 a partir de la loi jointe 2D.
    C'est la convolution ponderee par la dependance.
    """
    nx, ny = joint_2d.shape
    global_pmf = {}
    for i in range(nx):
        for j in range(ny):
            s = i + j
            global_pmf[s] = global_pmf.get(s, 0.0) + float(joint_2d[i, j])
    total = sum(global_pmf.values())
    if total <= 0:
        raise ValueError("Loi globale vide.")
    return {k: v / total for k, v in global_pmf.items()}
