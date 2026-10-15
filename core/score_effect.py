"""
Effet score : utilise les cotes 1X2 pour ponderer l'intensite 2e mi-temps.
Reference : MULTICORNER 360 v2, section 4, point 5.
"""
import numpy as np


RATIO_MIN = 0.70
RATIO_MAX = 1.50
FACTEUR_MIN = 0.75
FACTEUR_MAX = 1.35
SHRINKAGE_K = 5.0


def retirer_marge_1x2(o_home, o_draw, o_away):
    """Retrait de marge proportionnel sur les 3 cotes 1X2."""
    p_h = 1.0 / o_home
    p_d = 1.0 / o_draw
    p_a = 1.0 / o_away
    total = p_h + p_d + p_a
    return {
        "p_home": p_h / total,
        "p_draw": p_d / total,
        "p_away": p_a / total,
        "marge": total - 1.0,
    }


def apprendre_transition_ht_ft(hist):
    """
    P(HT | FT) avec Laplace smoothing (add-1).
    Perspective equipe : V = elle gagne/mene, N = nul, D = elle perd/est menee.
    Retourne matrice 3x3 : lignes = FT, colonnes = HT.
    """
    idx = {"V": 0, "N": 1, "D": 2}
    M = np.ones((3, 3))
    for m in hist:
        i = idx.get(m.get("resultat_final"))
        j = idx.get(m.get("resultat_mt"))
        if i is not None and j is not None:
            M[i, j] += 1.0
    M = M / M.sum(axis=1, keepdims=True)
    return M


def proba_ht_equipe(p_ft_v, p_ft_n, p_ft_d, M):
    """Proba d'etre en tete / nul / mene au HT, sachant les probas FT."""
    p_ft = np.array([p_ft_v, p_ft_n, p_ft_d])
    p_ht = p_ft @ M
    return {
        "p_leads_ht": float(p_ht[0]),
        "p_draw_ht": float(p_ht[1]),
        "p_trails_ht": float(p_ht[2]),
    }


def apprendre_ratios_2h(hist_home, hist_away):
    """
    Ratios E[c2h | scenario HT] / E[c2h] sur les 30 matchs combines.
    Shrinkage fort vers 1.0 pour ne pas surreagir au bruit.
    """
    c2h = [m["corners_2h"] for m in hist_home + hist_away]
    rmt = [m["resultat_mt"] for m in hist_home + hist_away]
    moy = float(np.mean(c2h))
    if moy <= 0:
        return {"V": 1.0, "N": 1.0, "D": 1.0}, {}
    ratios = {}
    stats = {}
    for s in ("V", "N", "D"):
        vals = [c2h[i] for i, r in enumerate(rmt) if r == s]
        n_s = len(vals)
        if n_s >= 3:
            r_obs = float(np.mean(vals)) / moy
            w = n_s / (n_s + SHRINKAGE_K)
            r = 1.0 + w * (r_obs - 1.0)
        else:
            r_obs = 1.0
            r = 1.0
        ratios[s] = max(RATIO_MIN, min(RATIO_MAX, r))
        stats[s] = {"n": n_s, "r_obs": r_obs, "r_ajuste": ratios[s]}
    return ratios, stats


def facteur_effet_score(p_leads, p_draw, p_trails, ratios):
    """Facteur multiplicatif a appliquer a l'intensite 2e MT."""
    facteur = (
        p_leads * ratios["V"]
        + p_draw * ratios["N"]
        + p_trails * ratios["D"]
    )
    return max(FACTEUR_MIN, min(FACTEUR_MAX, facteur))


def calculer_facteurs_match(hist_home, hist_away, odds_1x2):
    """
    Pipeline complet : facteurs 2e MT pour les deux equipes.
    """
    probs = retirer_marge_1x2(
        odds_1x2["home"], odds_1x2["draw"], odds_1x2["away"]
    )

    M_home = apprendre_transition_ht_ft(hist_home)
    M_away = apprendre_transition_ht_ft(hist_away)

    # Proba HT pour l'equipe domicile (V = elle mene)
    ht_home = proba_ht_equipe(
        probs["p_home"], probs["p_draw"], probs["p_away"], M_home
    )
    # Proba HT pour l'equipe exterieure (V = ELLE mene = p_away gagne)
    ht_away = proba_ht_equipe(
        probs["p_away"], probs["p_draw"], probs["p_home"], M_away
    )

    ratios, stats_ratios = apprendre_ratios_2h(hist_home, hist_away)

    f_home = facteur_effet_score(
        ht_home["p_leads_ht"], ht_home["p_draw_ht"], ht_home["p_trails_ht"],
        ratios
    )
    f_away = facteur_effet_score(
        ht_away["p_leads_ht"], ht_away["p_draw_ht"], ht_away["p_trails_ht"],
        ratios
    )

    return {
        "probs_ft": probs,
        "ht_home": ht_home,
        "ht_away": ht_away,
        "ratios": ratios,
        "stats_ratios": stats_ratios,
        "facteur_home_2h": f_home,
        "facteur_away_2h": f_away,
  }
