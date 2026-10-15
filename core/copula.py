"""
Copule de Frank pour modeliser la dependance entre 1re et 2e mi-temps.
Estimation du parametre theta par inversion du tau de Kendall.
Reference : MULTICORNER 360 v2, section 4, point 4.
"""
import numpy as np
from scipy.integrate import quad
from scipy.stats import kendalltau
from core.product_law import normalize


THETA_MIN = -30.0
THETA_MAX = 30.0
TAU_SEUIL = 0.05  # en dessous, on considere l'independance


def _C_frank(u, v, theta):
    """
    Fonction copule de Frank : C(u, v; theta).
    Gere u ou v = 0 ou 1, et theta proche de 0.
    """
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
    Loi jointe P(X=x, Y=y) a partir des marginales et d'une copule de Frank.
    Formule des differences finies :
      P(X=x, Y=y) = C(F(x), G(y)) - C(F(x-1), G(y))
                    - C(F(x), G(y-1)) + C(F(x-1), G(y-1))
    """
    px = np.asarray(pmf_x, dtype=float)
    py = np.asarray(pmf_y, dtype=float)
    px = px / px.sum()
    py = py / py.sum()

    fx = np.cumsum(px)  # CDF marginale de X
    fy = np.cumsum(py)  # CDF marginale de Y
    fx_prev = np.concatenate([[0.0], fx[:-1]])  # F_X(x-1)
    fy_prev = np.concatenate([[0.0], fy[:-1]])  # F_Y(y-1)

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
    joint = np.clip(joint, 0.0, None)  # corrige les petits negatifs numeriques
    return normalize(joint)


def _D1_debye(theta):
    """Premiere fonction de Debye : D1(theta) = (1/theta) * int_0^theta t/(e^t-1) dt."""
    if abs(theta) < 1e-8:
        return 1.0
    f = lambda t: t / (np.exp(t) - 1.0) if abs(t) > 1e-12 else 1.0
    val, _ = quad(f, 0.0, theta, limit=100)
    return float(val / theta)


def tau_from_theta(theta):
    """Tau de Kendall theorique pour une copule de Frank de parametre theta."""
    if abs(theta) < 1e-8:
        return 0.0
    return float(1.0 - 4.0 / theta + 4.0 / theta * _D1_debye(theta))


def theta_from_tau(tau):
    """
    Inverse tau -> theta par bissection sur [-30, 30].
    Monotone croissant.
    """
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
    """
    Estime theta a partir des observations brutes.
    Utilise tau-b (gere les ex-aequo) puis inversion.
    Retourne (theta, tau_obs, independance_ok).
    """
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
    """
    Construit la joint en estimant theta sur les observations.
    Si la dependance n'est pas distinguable de zero, retourne l'independance.
    """
    theta, tau, indep_ok = estimer_theta(x_obs, y_obs)
    if indep_ok:
        joint = normalize(np.outer(pmf_x, pmf_y))
    else:
        joint = pmf_joint_frank(pmf_x, pmf_y, theta)
    return joint, {"theta": theta, "tau": tau, "independance_ok": indep_ok}
