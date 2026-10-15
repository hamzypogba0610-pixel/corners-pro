"""
Module Pression : apprend la relation entre xG/xGoT et corners.
Utilise une regression lineaire multiple (moindres carres ordinaires).
Puis applique un ajustement multiplicatif a l'intensite de corners prevue.
Reference : MULTICORNER 360 v2, section 4 (couche contextuelle).
"""
import numpy as np


FEATURES = ("xg_pour", "xg_contre", "xgot_pour", "xgot_contre")
N_FEATURES = len(FEATURES)
REGULARISATION_MIN = 1e-6
FACTEUR_CLIP = (0.75, 1.35)  # on borne pour eviter les derives extremes


def _matrice_X(hist):
    """Matrice de design (n x (N_FEATURES + 1)) avec colonne de 1 pour beta0."""
    n = len(hist)
    X = np.ones((n, N_FEATURES + 1))
    for i, m in enumerate(hist):
        for j, f in enumerate(FEATURES):
            X[i, j + 1] = float(m[f])
    return X


def _vecteur_y(hist, cible):
    """Extrait les corners comme vecteur cible."""
    return np.array([float(m[cible]) for m in hist])


def fit_regression(hist, cible="corners_1h"):
    """
    Apprend beta par OLS avec regularisation ridge minimale.
    Retourne (beta, info) ou info contient r2, n, et la moyenne de y.
    """
    if len(hist) < N_FEATURES + 2:
        raise ValueError(
            f"Pas assez de matchs pour ajuster ({len(hist)} < {N_FEATURES + 2})"
        )
    X = _matrice_X(hist)
    y = _vecteur_y(hist, cible)

    # OLS ridge : beta = (X^T X + lambda I)^-1 X^T y
    # lambda minimal pour eviter les matrices singulieres
    XtX = X.T @ X
    n_features = XtX.shape[0]
    lam = REGULARISATION_MIN * np.trace(XtX) / n_features
    XtX_reg = XtX + lam * np.eye(n_features)
    Xty = X.T @ y
    beta = np.linalg.solve(XtX_reg, Xty)

    # R2
    y_pred = X @ beta
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return beta, {
        "r2": r2,
        "n": len(hist),
        "y_moyenne": float(np.mean(y)),
        "y_std": float(np.std(y)),
        "beta": beta.tolist(),
        "features": list(FEATURES),
    }


def predict_corners(hist_mean_features, beta):
    """
    Predit les corners a partir d'un vecteur MOYEN de features.
    hist_mean_features : dict {xg_pour: ..., xg_contre: ..., ...}
    """
    x = np.ones(N_FEATURES + 1)
    for j, f in enumerate(FEATURES):
        x[j + 1] = float(hist_mean_features[f])
    return float(x @ beta)


def moyenne_features(hist):
    """Moyenne de chaque feature sur l'historique."""
    return {
        f: float(np.mean([m[f] for m in hist]))
        for f in FEATURES
    }


def facteur_ajustement(hist_equipe, beta_1h, beta_2h):
    """
    Calcule un facteur multiplicatif a appliquer aux intensites de corners.
    Principe :
      - On predit les corners attendus a partir des xG moyens (modele pression)
      - On calcule aussi la moyenne brute des corners observes
      - Le ratio predit/observe donne un facteur de correction,
        borne dans FACTEUR_CLIP pour eviter les derives.
    """
    moy_feat = moyenne_features(hist_equipe)
    pred_1h = predict_corners(moy_feat, beta_1h)
    pred_2h = predict_corners(moy_feat, beta_2h)

    obs_1h = float(np.mean([m["corners_1h"] for m in hist_equipe]))
    obs_2h = float(np.mean([m["corners_2h"] for m in hist_equipe]))

    # Ratio predit / observe ; si observe = 0, on renvoie 1.0 (neutre)
    r1 = pred_1h / obs_1h if obs_1h > 0 else 1.0
    r2 = pred_2h / obs_2h if obs_2h > 0 else 1.0

    # Clipping pour eviter les extremes
    r1 = max(FACTEUR_CLIP[0], min(FACTEUR_CLIP[1], r1))
    r2 = max(FACTEUR_CLIP[0], min(FACTEUR_CLIP[1], r2))

    return {
        "facteur_1h": r1,
        "facteur_2h": r2,
        "pred_1h": pred_1h,
        "pred_2h": pred_2h,
        "obs_1h": obs_1h,
        "obs_2h": obs_2h,
        "moy_feat": moy_feat,
    }


def entrainer_sur_les_deux_equipes(hist_home, hist_away):
    """
    Entraine les regressions sur la CONCATENATION des deux historiques.
    Pourquoi ? Parce qu'on veut apprendre une relation xG->corners qui
    soit valable pour les deux equipes et capte une realite de la ligue.
    """
    tout = hist_home + hist_away
    beta_1h, info_1h = fit_regression(tout, cible="corners_1h")
    beta_2h, info_2h = fit_regression(tout, cible="corners_2h")
    return {
        "beta_1h": beta_1h,
        "beta_2h": beta_2h,
        "info_1h": info_1h,
        "info_2h": info_2h,
}
