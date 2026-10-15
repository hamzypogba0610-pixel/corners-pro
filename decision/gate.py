"""
Porte de decision : applique les 6 criteres de la section 8.
Par defaut, retourne un refus. Une selection doit etre meritee.
Reference : MULTICORNER 360 v2, sections 8 et 9.
"""
from decision.ev import edge_vs_implicite, penalite_winner_curse


N_MIN = 15
EDGE_MIN = 0.04
SPREAD_MAX = 0.08
EV_P20_MIN = 0.0


CRITERES_DESCRIPTIONS = {
    1: "n >= 15 matchs exploitables par equipe",
    2: "edge vs implicite sans marge >= 4 points",
    3: "EV au 20e percentile > 0",
    4: "desaccord entre modeles < 8 points",
    5: "recommandation stable en sensibilite",
    6: "aucun faux signal majeur",
}


def detecter_faux_signaux(contexte):
    """
    Cherche activement les arguments CONTRE (section 9).
    Seuils calibres pour ne PAS bloquer des cas normaux.
    """
    alertes = []

    # Plus de 2 cartons rouges parmi les 5 derniers matchs d'une equipe
    if contexte.get("cartons_rouges_recents", 0) >= 2:
        alertes.append(
            "historique recent contient plusieurs cartons rouges (>=2 sur 5 matchs)"
        )

    if contexte.get("changement_entraineur", False):
        alertes.append("changement d'entraineur signalé")

    if contexte.get("cv_mc", 0.0) > 2.0:
        alertes.append("coefficient de variation de MC extrême (>2.0)")

    # Zero tres frequent rend le marche peu fiable
    if contexte.get("freq_zero_mc", 0.0) > 0.5:
        alertes.append("fréquence de zéro sur MC supérieure à 50%")

    if contexte.get("cote_min", 10.0) < 1.30:
        alertes.append("cote trop basse pour le risque pris (< 1.30)")

    if not contexte.get("pagerank_homogene", True):
        alertes.append("adversaires non comparables (PageRank dispersé)")

    if contexte.get("rotation_probable", False):
        alertes.append("rotation probable (enjeu faible)")

    return alertes


def _verifier_critere_1(n):
    return n >= N_MIN, f"n = {n} (min {N_MIN})"


def _verifier_critere_2(edge):
    return edge >= EDGE_MIN, f"edge = {edge*100:.2f} pts (min {EDGE_MIN*100:.0f})"


def _verifier_critere_3(ev_p20):
    return ev_p20 > EV_P20_MIN, f"EV_p20 = {ev_p20:.4f}"


def _verifier_critere_4(spread):
    return spread < SPREAD_MAX, f"spread = {spread*100:.2f} pts"


def _verifier_critere_5(stable):
    return bool(stable), f"stable = {stable}"


def _verifier_critere_6(faux_signaux):
    if not faux_signaux:
        return True, "aucun"
    return False, " | ".join(faux_signaux)


def evaluer_selection(analyse):
    """
    Analyse : dict avec toutes les infos necessaires.
    Retourne dict avec accepte, raisons_refus, detail_criteres.
    """
    p_est = float(analyse["p_estime"])
    p_imp = float(analyse["p_implicite_sans_marge"])

    p_est_corrige = penalite_winner_curse(p_est)
    edge = edge_vs_implicite(p_est_corrige, p_imp)

    faux_signaux = detecter_faux_signaux(analyse.get("contexte", {}))

    criteres = {}
    criteres[1] = _verifier_critere_1(int(analyse["n"]))
    criteres[2] = _verifier_critere_2(edge)
    criteres[3] = _verifier_critere_3(float(analyse["ev_p20"]))
    criteres[4] = _verifier_critere_4(float(analyse.get("spread_modeles", 0.0)))
    criteres[5] = _verifier_critere_5(analyse.get("stable_sensibilite", True))
    criteres[6] = _verifier_critere_6(faux_signaux)

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
