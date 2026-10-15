"""
PageRank pondere pour evaluer la difficulte des adversaires.
Chaque match recoit un poids base sur la force de l'adversaire rencontre.
Reference : MULTICORNER 360 v2, section 4, point 5 (couche algorithmique).
"""
import numpy as np


DUMPING = 0.85
MAX_ITER = 100
TOL = 1e-9
POIDS_MIN = 0.5
POIDS_MAX = 1.5


def construire_graphe(hist, equipe_cible):
    """
    Construit un graphe oriente a partir de l'historique.
    Noeuds : equipe_cible + tous les adversaires.
    Arete equipe -> adversaire, poids = corners_1h + corners_2h de l'equipe
    contre cet adversaire ce match-la.
    Retourne (noeuds, index, edges, poids_aretes).
    """
    noeuds = [equipe_cible]
    index = {equipe_cible: 0}
    edges = []
    poids = []
    for m in hist:
        opp = m["opponent"]
        if opp not in index:
            index[opp] = len(noeuds)
            noeuds.append(opp)
        c = int(m.get("corners_1h", 0)) + int(m.get("corners_2h", 0))
        edges.append((index[equipe_cible], index[opp], c))
        poids.append(c)
    return noeuds, index, edges, poids


def pagerank(noeuds, edges, dumping=DUMPING, max_iter=MAX_ITER, tol=TOL):
    """
    PageRank pondere.
    edges : liste de tuples (i, j, w) -> arete i -> j de poids w.
    Retourne un vecteur de scores (un par noeud), sommant a 1.
    """
    n = len(noeuds)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])

    # Construire la matrice de transition M : M[j, i] = w(i->j) / sum_k w(i->k)
    # (colonne i = distribution de sortie du noeud i)
    M = np.zeros((n, n))
    for i, j, w in edges:
        M[j, i] += float(w)

    # Normaliser chaque colonne
    for i in range(n):
        s = M[:, i].sum()
        if s > 0:
            M[:, i] /= s
        else:
            # Noeud sans sortie : on redistribue uniformement
            M[:, i] = 1.0 / n

    # PageRank standard
    v = np.ones(n) / n
    for _ in range(max_iter):
        v_new = dumping * M @ v + (1.0 - dumping) / n
        if np.sum(np.abs(v_new - v)) < tol:
            break
        v = v_new
    v = v_new
    v = v / v.sum()  # securite numerique
    return v


def calculer_poids_matchs(hist, equipe_cible, scores=None):
    """
    Donne un poids a chaque match de l'historique, base sur la difficulte
    de l'adversaire (via PageRank).
    Retourne une liste de poids (meme ordre que hist), bornes dans [0.5, 1.5].
    """
    noeuds, index, edges, _ = construire_graphe(hist, equipe_cible)
    if scores is None:
        scores = pagerank(noeuds, edges)

    poids = []
    for m in hist:
        opp_idx = index[m["opponent"]]
        s = scores[opp_idx] if len(scores) > opp_idx else 0.0
        poids.append(s)

    arr = np.array(poids, dtype=float)
    moy = arr.mean() if arr.size > 0 else 1.0
    if moy <= 0:
        return [1.0] * len(hist)

    # Normaliser autour de 1 : poids_brut / moyenne
    rel = arr / moy

    # Clip pour eviter les extremes
    rel = np.clip(rel, POIDS_MIN, POIDS_MAX)
    return rel.tolist()


def rapport_pagerank(hist_home, hist_away, home_team, away_team):
    """
    Rapport complet : poids des matchs pour les deux equipes.
    """
    poids_h = calculer_poids_matchs(hist_home, home_team)
    poids_a = calculer_poids_matchs(hist_away, away_team)
    return {
        "poids_home": poids_h,
        "poids_away": poids_a,
        "moyenne_home": float(np.mean(poids_h)),
        "moyenne_away": float(np.mean(poids_a)),
        "std_home": float(np.std(poids_h)),
        "std_away": float(np.std(poids_a)),
        "homogene": (
            float(np.std(poids_h)) < 0.15 and float(np.std(poids_a)) < 0.15
        ),
  }
