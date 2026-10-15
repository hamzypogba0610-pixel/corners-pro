"""
Porte de decision : applique les 6 criteres de la section 8.
Par defaut, retourne un refus. Une selection doit etre meritee.
Reference : MULTICORNER 360 v2, sections 8 et 9.
"""
from decision.ev import edge_vs_implicite, penalite_winner_curse


N_MIN = 15
EDGE_MIN = 0.04             # 4 points de proba
SPREAD_MAX = 0.08           # 8 points
EV_P20_MIN = 0.0
SEUIL_TORNADO = 0.02        # variation max toleree en sensibilite


CRITERES_DESCRIPTIONS = {
    1: "n >= 15 matchs exploitables par equipe",
    2: "edge vs implicite sans marge >= 4 points",
    3: "EV au 20e percentile > 0",
    4: "desaccord entre modeles < 8 points",
    5: "recommandation stable en sensibilite",
    6: "aucun faux signal majeur",
}


def _verifier_critere_1(n):
    return n >= N_MIN, f"n = {n} (min {N_MIN})"


def _verifier_critere_2(p_estime, p_implicite_sans_marge):
    edge = edge_vs_implicite(p_estime, p_implicite_sans_marge)
    return edge >= EDGE_MIN, f"edge = {edge*100:.2f} pts (min {EDGE_MIN*100:.0f})"


def _verifier_critere_3(ev_p20):
    return ev_p20 > EV_P20_MIN, f"EV_p20 = {ev_p20:.4f}"


def _verifier_critere_4(spread_modeles):
    return spread_modeles < SPREAD_MAX, f"spread = {spread_modeles*100:.2f} pts"


def _verifier_critere_5(stable_sensibilite):
    return bool(stable_sensibilite), f"stable = {stable_sensibilite}"


def _verifier_critere_6(faux_signaux):
    """faux_signaux : liste de chaines detaillant les problemes detectes."""
    return len(faux_signaux) == 0, f"{len(faux_signaux)} faux signal(aux)"


def detecter_faux_signaux(contexte):
    """
    Avant toute selection, chercher activement les arguments CONTRE.
    Section 9 du cahier des charges.
    contexte : dict avec les infos disponibles.
    Retourne une liste de chaines (vide si aucun probleme majeur).
    """
    alertes = []

    # Cartons rouges dans l'historique proche = matchs atypiques
    if contexte.get("cartons_rouges_recents", 0) >= 2:
        alertes.append("historique contient plusieurs cartons rouges recents")

    # Equipes tres differentes entre hist et match analyse
    if contexte.get("changement_entraineur", False):
        alertes.append("changement d'entraineur signalé")

    # Donnees trop volatiles
    if contexte.get("cv_mc", 0.0) > 1.5:
        alertes.append("coefficient de variation de MC trop élevé")

    # Zero très frequent = marche peu fiable
    if contexte.get("freq_zero_mc", 0.0) > 0.4:
        alertes.append("fréquence de zéro sur MC supérieure à 40%")

    # Cote trop basse pour le risque
    if contexte.get("cote_min", 10.0) < 1.30:
        alertes.append("cote trop basse pour le risque pris")

    # Adversaires non comparables (PageRank très dispersé)
    if not contexte.get("pagerank_homogene", True):
        alertes.append("adversaires non comparables (PageRank dispersé)")

    # Match à enjeu faible ou rotation probable
    if contexte.get("rotation_probable", False):
        alertes.append("rotation probable (enjeu faible)")

    return alertes


def evaluer_selection(analyse):
    """
    Analyse : dict avec toutes les infos necessaires.
    Cles attendues :
      - n : int, taille echantillon exploitable
      - p_estime : float, probabilite estimee par le modele
      - p_implicite_sans_marge : float, proba bookmaker apres retrait marge
      - ev_p20 : float, EV au 20e percentile
      - spread_modeles : float (defaut 0.0 si non calcule)
      - stable_sensibilite : bool (defaut True si non teste)
      - contexte : dict pour detecter_faux_signaux
    Retourne dict : {
        "accepte": bool,
        "raisons_refus": [str],
        "detail_criteres": {1..6: {"ok": bool, "detail": str}},
        "faux_signaux": [str]
    }
    """
    p_est = float(analyse["p_estime"])
    p_imp = float(analyse["p_implicite_sans_marge"])

    # Correction winner's curse AVANT comparaison
    p_est_corrige = penalite_winner_curse(p_est)
    edge = edge_vs_implicite(p_est_corrige, p_imp)

    faux_signaux = detecter_faux_signaux(analyse.get("contexte", {}))

    criteres = {}
    criteres[1] = _verifier_critere_1(int(analyse["n"]))
    criteres[2] = (edge >= EDGE_MIN,
                   f"edge corrigé (winner's curse) = {edge*100:.2f} pts")
    criteres[3] = _verifier_critere_3(float(analyse["ev_p20"]))
    criteres[4] = _verifier_critere_4(float(analyse.get("spread_modeles", 0.0)))
    criteres[5] = _verifier_critere_5(analyse.get("stable_sensibilite", True))
    criteres[6] = _verifier_critere_6(faux_signaux)

    detail = {
        i: {"ok": ok, "detail": d, "description": CRITERES_DESCRIPTIONS[i]}
        for i, (ok, d) in criteres.items()
    }

    refus = [f"Critere {i} : {info['detail']}"
             for i, info in detail.items() if not info["ok"]]

    return {
        "accepte": len(refus) == 0,
        "raisons_refus": refus,
        "detail_criteres": detail,
        "faux_signaux": faux_signaux,
        "p_estime_corrige": p_est_corrige,
        "edge_corrige": edge,
    }


def verdict_texte(resultat):
    """Rend un verdict lisible."""
    if resultat["accepte"]:
        return "SELECTION RETENUE"
    return "AUCUNE SELECTION SUFFISAMMENT SOLIDE POUR CE MARCHE."
