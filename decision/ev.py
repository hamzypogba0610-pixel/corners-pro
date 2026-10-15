"""
Calculs d'EV (esperance de valeur) et Kelly fractionnaire.
Reference : MULTICORNER 360 v2, section 8.
"""
import numpy as np


PENALITE_WINNER_CURSE = 0.02  # 2 points de proba retires
KELY_PLAFOND = 0.25           # jamais plus d'un quart de Kelly


def ev_over(p_gain, cote_over):
    """EV d'un pari Over : p*(cote-1) - (1-p)."""
    if not 0.0 <= p_gain <= 1.0:
        raise ValueError(f"p_gain hors [0,1] : {p_gain}")
    if cote_over <= 1.0:
        raise ValueError(f"cote invalide : {cote_over}")
    return p_gain * (cote_over - 1.0) - (1.0 - p_gain)


def ev_under(p_gain, cote_under):
    return ev_over(p_gain, cote_under)


def kelly_fraction(p_gain, cote):
    b = cote - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - p_gain
    f = (p_gain * b - q) / b
    return max(0.0, min(1.0, f))


def kelly_prudent(p_gain, cote, plafond=KELY_PLAFOND):
    f = kelly_fraction(p_gain, cote)
    return min(f, plafond)


def ev_posterior(p_samples, cote):
    """
    EV calcule sur la distribution a posteriori de p.
    Retourne : EV mediane, EV 20e percentile, P(EV>0), ET
    un intervalle de credibilite a 80 % sur la probabilite.
    """
    p = np.asarray(p_samples, dtype=float)
    if p.size == 0:
        raise ValueError("Aucun echantillon a posteriori.")
    evs = p * (cote - 1.0) - (1.0 - p)
    return {
        "ev_mediane": float(np.median(evs)),
        "ev_moyenne": float(np.mean(evs)),
        "ev_p20": float(np.percentile(evs, 20)),
        "ev_p80": float(np.percentile(evs, 80)),
        "p_ev_positif": float(np.mean(evs > 0)),
        "p_gain_mediane": float(np.median(p)),
        "p_gain_p20": float(np.percentile(p, 20)),
        "p_gain_p10": float(np.percentile(p, 10)),
        "p_gain_p90": float(np.percentile(p, 90)),
        "ic80_bas": float(np.percentile(p, 10)),
        "ic80_haut": float(np.percentile(p, 90)),
        "ic80_largeur": float(
            np.percentile(p, 90) - np.percentile(p, 10)
        ),
    }


def edge_vs_implicite(p_estime, p_implicite_sans_marge):
    return float(p_estime - p_implicite_sans_marge)


def penalite_winner_curse(p_estime):
    return max(0.0, p_estime - PENALITE_WINNER_CURSE)
