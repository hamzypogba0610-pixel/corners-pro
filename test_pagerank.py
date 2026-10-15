"""Tests du PageRank pondere."""
import numpy as np
from core.pagerank import (
    construire_graphe, pagerank, calculer_poids_matchs, rapport_pagerank,
)


def hist_bidon(adversaires, corners=3):
    return [
        {"date": f"2026-01-{i+1:02d}", "opponent": opp, "venue": "H",
         "corners_1h": corners, "corners_2h": corners,
         "resultat_final": "V", "resultat_mt": "N", "cartons_rouges": 0,
         "xg_pour": 1.5, "xg_contre": 0.9,
         "xgot_pour": 1.2, "xgot_contre": 0.7}
        for i, opp in enumerate(adversaires)
    ]


def test_graphe_construction():
    h = hist_bidon(["A", "B", "C", "A", "B"])
    noeuds, index, edges, poids = construire_graphe(h, "MonEquipe")
    # noeuds : MonEquipe + A + B + C = 4
    assert len(noeuds) == 4
    assert "MonEquipe" in index
    assert "A" in index
    assert len(edges) == 5
    print("  OK : graphe construit correctement")


def test_pagerank_somme_a_un():
    h = hist_bidon(["A", "B", "C"])
    noeuds, index, edges, _ = construire_graphe(h, "MonEquipe")
    scores = pagerank(noeuds, edges)
    assert len(scores) == len(noeuds)
    assert abs(scores.sum() - 1.0) < 1e-8
    print(f"  OK : PageRank somme a 1, scores={[round(s,3) for s in scores]}")


def test_pagerank_deterministe():
    """Deux appels doivent donner le meme resultat."""
    h = hist_bidon(["A", "B", "A", "B", "C"])
    noeuds, _, edges, _ = construire_graphe(h, "X")
    s1 = pagerank(noeuds, edges)
    s2 = pagerank(noeuds, edges)
    assert np.allclose(s1, s2)
    print("  OK : PageRank deterministe")


def test_poids_dans_bornes():
    h = hist_bidon(["A", "B", "C", "D"])
    poids = calculer_poids_matchs(h, "X")
    assert all(0.5 <= p <= 1.5 for p in poids)
    print(f"  OK : poids bornes : {[round(p, 3) for p in poids]}")


def test_poids_moyenne_proche_de_un():
    """Par construction, la moyenne des poids doit etre proche de 1."""
    h = hist_bidon(["A", "B", "C", "D", "E"])
    poids = calculer_poids_matchs(h, "X")
    moy = np.mean(poids)
    assert abs(moy - 1.0) < 0.3
    print(f"  OK : moyenne des poids = {moy:.3f}")


def test_rapport_complet():
    h1 = hist_bidon(["A", "B", "C"])
    h2 = hist_bidon(["D", "E", "F"])
    r = rapport_pagerank(h1, h2, "Home", "Away")
    assert "poids_home" in r and "poids_away" in r
    assert len(r["poids_home"]) == 3
    assert "homogene" in r
    print(f"  OK : rapport complet, homogene={r['homogene']}")


if __name__ == "__main__":
    tests = [
        test_graphe_construction,
        test_pagerank_somme_a_un,
        test_pagerank_deterministe,
        test_poids_dans_bornes,
        test_poids_moyenne_proche_de_un,
        test_rapport_complet,
    ]
    print("=== Tests du PageRank ===\n")
    r = 0
    for t in tests:
        try:
            t(); r += 1
        except Exception as e:
            print(f"  ECHEC : {t.__name__} -> {e}")
    print(f"\n=== {r}/{len(tests)} tests reussis ===")
    if r != len(tests):
        raise SystemExit(1)
