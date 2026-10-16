"""
Test de sensibilite (tableau tornade).
Fait varier les parametres du modele pour verifier la robustesse
de la decision.
Reference : MULTICORNER 360 v2, section 7.

Note performance : la tornade relance le pipeline complet pour chaque
scenario. Sur Render free (0.1 CPU), 7 scenarios depassent le timeout
de 100 secondes. On en garde 3 (les plus utiles) pour rester dans les
limites. A re-activer si on migre sur un serveur plus rapide.
"""
import numpy as np


# Tornade reduite a 3 scenarios.
# Retires : demi_vie_courte, demi_vie_longue (peu discriminants),
# independance_MC, dependance_forte (relancent la copule 2D, tres lourds).
SCENARIOS = [
    ("moyenne_basse",   {"facteur_moyenne": 0.85}),
    ("moyenne_haute",   {"facteur_moyenne": 1.15}),
    ("sans_extremes",   {"retirer_extremes": 2}),
]


def perturber_hist(hist, facteur_moyenne=1.0, retirer_extremes=0):
    """
    Cree une version perturbee de l'historique.
    facteur_moyenne : multiplication des corners (defaut 1.0).
    retirer_extremes : nb de paires extremes (min et max) a remplacer
                        par la mediane. Garde n=15.
    """
    h = [dict(m) for m in hist]
    if facteur_moyenne != 1.0:
        for m in h:
            m["corners_1h"] = max(
                0, int(round(m["corners_1h"] * facteur_moyenne))
            )
            m["corners_2h"] = max(
                0, int(round(m["corners_2h"] * facteur_moyenne))
            )
    if retirer_extremes > 0 and len(h) > 2 * retirer_extremes + 1:
        mcs = [(m["corners_1h"] * m["corners_2h"], i)
               for i, m in enumerate(h)]
        mcs_sorted = sorted(mcs)
        to_fix = set()
        for k in range(retirer_extremes):
            to_fix.add(mcs_sorted[k][1])
            to_fix.add(mcs_sorted[-1 - k][1])
        c1h_all = sorted([m["corners_1h"] for m in h])
        c2h_all = sorted([m["corners_2h"] for m in h])
        med_1h = c1h_all[len(c1h_all) // 2]
        med_2h = c2h_all[len(c2h_all) // 2]
        for i in to_fix:
            h[i]["corners_1h"] = med_1h
            h[i]["corners_2h"] = med_2h
    return h
