"""
Porte de decision : applique les 6 criteres + IC 80%.
Reference : MULTICORNER 360 v2, sections 7 et 8.
"""
from decision.ev import edge_vs_implicite, penalite_winner_curse


N_MIN = 15
EDGE_MIN = 0.04
SPREAD_MAX = 0.08
EV_P20_MIN = 0.0
IC80_LARGEUR_MAX = 0.20  # si l'intervalle a 80% depasse 20 pts -> refus


CRITERES_DESCRIPTIONS = {
    1: "n >= 15 matchs exploitables par equipe",
    2: "edge vs implicite sans marge >= 4 points",
    3: "EV au 20e percentile > 0",
    4: "desaccord entre modeles < 8 points",
    5: "recommandation stable en sensibilite",
    6: "aucun faux signal majeur",
    7: "intervalle de credibilite 80% pas trop large",
}


def detecter_faux_signaux(contexte):
    alertes = []

    if contexte.get("cartons_rouges_recents", 0) >= 2:
        alertes.append(
            "historique récent contient plusieurs cartons rouges (>=2 sur 5 matchs)"
        )
    if contexte.get("changement_entraineur", False):
        alertes.append("changement d'entraineur signalé")
    if contexte.get("cv_mc", 0.0) > 2.0:
        alertes.append("coefficient de variation de MC extrême (>2.0)")
    if contexte.get("freq_zero_mc", 0.0) > 0.5:
        alertes.append("fréquence de zéro sur MC supérieure à 50%")
    if contexte.get("cote_min", 10.0) < 1.30:
        alertes.append("cote trop basse pour le risque pris (< 1.30)")
    if not contexte.get("pagerank_homogene", True):
        alertes.append("adversaires non comparables (PageRank dispersé)")
    if contexte.get("rotation_probable", False):
        alertes.append("rotation probable (enjeu faible)")
    return alertes


def _c1(n):
    return n >= N_MIN, f"n = {n} (min {N_MIN})"

def _c2(edge):
    return edge >= EDGE_MIN, f"edge = {edge*100:.2f} pts (min {EDGE_MIN*100:.0f})"

def _c3(ev_p20):
    return ev_p20 > EV_P20_MIN, f"EV_p20 = {ev_p20:.4f}"

def _c4(spread):
    return spread < SPREAD_MAX, f"spread = {spread*100:.2f} pts"

def _c5(stable):
    return bool(stable), f"stable = {stable}"

def _c6(faux):
    if not faux:
        return True, "aucun"
    return False, " | ".join(faux)

def _c7(ic80_largeur):
    if ic80_largeur is None:
        return True, "non calcule (ignore)"
    return (
        ic80_largeur < IC80_LARGEUR_MAX,
        f"largeur IC80 = {ic80_largeur*100:.2f} pts "
        f"(max {IC80_LARGEUR_MAX*100:.0f})"
    )


def evaluer_selection(analyse):
    p_est = float(analyse["p_estime"])
    p_imp = float(analyse["p_implicite_sans_marge"])

    p_est_corrige = penalite_winner_curse(p_est)
    edge = edge_vs_implicite(p_est_corrige, p_imp)

    faux_signaux = detecter_faux_signaux(analyse.get("contexte", {}))

    criteres = {}
    criteres[1] = _c1(int(analyse["n"]))
    criteres[2] = _c2(edge)
    criteres[3] = _c3(float(analyse["ev_p20"]))
    criteres[4] = _c4(float(analyse.get("spread_modeles", 0.0)))
    criteres[5] = _c5(analyse.get("stable_sensibilite", True))
    criteres[6] = _c6(faux_signaux)
    criteres[7] = _c7(analyse.get("ic80_largeur", None))

    detail = {
        i: {"ok": ok, "detail": d, "description": CRITERES_DESCRIPTIONS[i]}
        for i, (ok, d) in criteres.items()
    }

    refus = [
        f"Critère {i} — {detail[i]['description']} : {detail[i]['detail']}"
        for i in detail if not detail[i]["ok"]
    ]

    return {
        "accepte": len(refus) == 0,
        "raisons_refus": refus,
        "detail_criteres": detail,
        "faux_signaux": faux_signaux,
        "p_estime_corrige": p_est_corrige,
        "edge_corrige": edge,
    }


def verdict_texte(resultat):
    if resultat["accepte"]:
        return "SELECTION RETENUE"
    return "AUCUNE SELECTION SUFFISAMMENT SOLIDE POUR CE MARCHE."
