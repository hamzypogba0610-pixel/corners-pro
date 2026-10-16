"""
Calibration des probabilites : Temperature, Platt, Beta + ensemble log-odds.
Reference : Platt 1999, Kull 2017, Gupta 2024.
Option A : moyenne des log-odds des 3 calibrateurs.
"""
import numpy as np


EPS = 1e-12


def _clip(p, eps=EPS):
    p = np.asarray(p, dtype=float)
    return np.clip(p, eps, 1.0 - eps)


def logit(p):
    p = _clip(p)
    return np.log(p / (1.0 - p))


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


# ---------------------------------------------------------------------------
# Temperature scaling (1 parametre)
# ---------------------------------------------------------------------------

def calibrer_temperature(p, T=1.0):
    if T <= 0:
        raise ValueError("T doit etre > 0")
    return float(sigmoid(logit(p) / T))


def fit_temperature(p_pred, y_reel):
    from scipy.optimize import minimize_scalar
    p = np.array(p_pred, dtype=float)
    y = np.array(y_reel, dtype=float)
    if len(p) < 5:
        return 1.0

    def loss(T):
        if T <= 0:
            return 1e10
        q = sigmoid(logit(p) / T)
        q = _clip(q)
        return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))

    res = minimize_scalar(loss, bounds=(0.1, 10.0), method="bounded")
    return float(res.x) if res.success else 1.0


# ---------------------------------------------------------------------------
# Platt scaling (2 parametres)
# ---------------------------------------------------------------------------

def calibrer_platt(p, a=1.0, b=0.0):
    return float(sigmoid(a * logit(p) + b))


def fit_platt(p_pred, y_reel):
    from scipy.optimize import minimize
    p = np.array(p_pred, dtype=float)
    y = np.array(y_reel, dtype=float)
    if len(p) < 10:
        return 1.0, 0.0

    def loss(params):
        a, b = params
        q = sigmoid(a * logit(p) + b)
        q = _clip(q)
        return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))

    res = minimize(loss, x0=[1.0, 0.0], method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 500})
    if res.success:
        return float(res.x[0]), float(res.x[1])
    return 1.0, 0.0


# ---------------------------------------------------------------------------
# Beta calibration (3 parametres)
# ---------------------------------------------------------------------------

def calibrer_beta(p, a=1.0, b=1.0, c=0.0):
    """
    Formule : sigmoid(c + a*log(p) - b*log(1-p)).
    Avec a=b=1, c=0 : identite.
    """
    p = _clip(p)
    return float(sigmoid(c + a * np.log(p) - b * np.log(1.0 - p)))


def fit_beta(p_pred, y_reel):
    from scipy.optimize import minimize
    p = np.array(p_pred, dtype=float)
    y = np.array(y_reel, dtype=float)
    if len(p) < 15:
        return 1.0, 1.0, 0.0

    def loss(params):
        a, b, c = params
        pc = _clip(p)
        q = sigmoid(c + a * np.log(pc) - b * np.log(1.0 - pc))
        q = _clip(q)
        return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))

    res = minimize(loss, x0=[1.0, 1.0, 0.0], method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 800})
    if res.success:
        return float(res.x[0]), float(res.x[1]), float(res.x[2])
    return 1.0, 1.0, 0.0


# ---------------------------------------------------------------------------
# Ensemble par moyenne des log-odds (Option A)
# ---------------------------------------------------------------------------

def combiner_ensembles(p, params=None):
    """
    Moyenne des log-odds des 3 calibrateurs.
    Si params est None/vide, on utilise les valeurs identite
    (T=1, a_platt=1, b_platt=0, a_beta=1, b_beta=1, c_beta=0) :
    l'ensemble retourne p inchange.
    """
    p = float(_clip(p))
    if not params:
        return p
    T = params.get("T", 1.0)
    a_p = params.get("a_platt", 1.0)
    b_p = params.get("b_platt", 0.0)
    a_b = params.get("a_beta", 1.0)
    b_b = params.get("b_beta", 1.0)
    c_b = params.get("c_beta", 0.0)
    q_temp = calibrer_temperature(p, T)
    q_platt = calibrer_platt(p, a_p, b_p)
    q_beta = calibrer_beta(p, a_b, b_b, c_b)
    z = (logit(q_temp) + logit(q_platt) + logit(q_beta)) / 3.0
    return float(sigmoid(z))


def entrainer_calibrateurs(p_pred, y_reel):
    """
    Entraine les 3 calibrateurs sur (p_pred, y_reel).
    Retourne dict des parametres. Utilise plus tard quand on aura
    accumule assez de donnees reelles.
    """
    T = fit_temperature(p_pred, y_reel)
    a_p, b_p = fit_platt(p_pred, y_reel)
    a_b, b_b, c_b = fit_beta(p_pred, y_reel)
    return {
        "T": T,
        "a_platt": a_p, "b_platt": b_p,
        "a_beta": a_b, "b_beta": b_b, "c_beta": c_b,
        "n": len(p_pred),
    }


def appliquer(p, params=None):
    """
    Point d'entree : applique la calibration.
    Retourne (p_calibree, info).
    """
    p = float(_clip(p))
    if not params:
        return p, {"calibree": False, "methode": "aucune"}
    p_cal = combiner_ensembles(p, params)
    return p_cal, {
        "calibree": True,
        "methode": "ensemble_log_odds",
        "params": params,
  }
