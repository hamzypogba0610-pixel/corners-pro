"""Tests du module effet score."""
import numpy as np
from core.score_effect import (
    retirer_marge_1x2, apprendre_transition_ht_ft,
    proba_ht_equipe, apprendre_ratios_2h,
    facteur_effet_score, calculer_facteurs_match,
    RATIO_MIN, RATIO_MAX, FACTEUR_MIN, FACTEUR_MAX,
)


def hist_simple(n=15, rf="V", rmt="V", c2h=4):
    return [
        {"date": f"2026-01-{i+1:02d}", "opponent": f"X{i}", "venue": "H",
         "corners_1h": 3, "corners_2h": c2h,
         "resultat_final": rf, "resultat_mt": rmt,
         "cartons_rouges": 0,
         "xg_pour": 1.5, "xg_contre": 0.9,
         "xgot_pour": 1.2, "xgot_contre": 0.7}
        for i in range(n)
    ]


def test_retirer_marge_somme_a_un():
    r = retirer_marge_1x2(2.10, 3.40, 3.20)
    s = r["p_home"] + r["p_draw"] + r["p_away"]
    assert abs(s - 1.0) < 1e-9
    assert r["marge"] > 0
    print(f"  OK : marge retirée {r['marge']*100:.2f}%, somme=1")


def test_retirer_marge_equilibre():
    r = retirer_marge_1x2(3.0, 3.0, 3.0)
    assert abs(r["p_home"] - 1/3) < 1e-9
    print("  OK : 3 cotes égales → 1/3 chacun")


def test_transition_somme_ligne_un():
    h = hist_simple(15, "V", "V")
    M = apprendre_transition_ht_ft(h)
    for i in range(3):
        assert abs(M[i].sum() - 1.0) < 1e-9
    print("  OK : matrice transition lignes somme=1")


def test_transition_smoothed():
    """Avec peu de données, la matrice doit rester loin de 0/1 (smoothing)."""
    h = hist_simple(3, "V", "V")  # seulement 3 matchs, tous V/V
    M = apprendre_transition_ht_ft(h)
    # Sans smoothing on aurait M[0] = [1, 0, 0]
    # Avec add-1 on doit avoir des valeurs > 0 partout
    assert M[0, 1] > 0
    assert M[0, 2] > 0
    print(f"  OK : smoothing actif, M[V]={M[0].round(3)}")


def test_proba_ht_equipe():
    """Une équipe qui gagne 100% du temps doit mener au HT plus souvent."""
    h = hist_simple(15, "V", "V")
    M = apprendre_transition_ht_ft(h)
    r = proba_ht_equipe(1.0, 0.0, 0.0, M)  # elle gagne toujours
    assert r["p_leads_ht"] > r["p_trails_ht"]
    print(f"  OK : p(leads)={r['p_leads_ht']:.3f} > p(trails)={r['p_trails_ht']:.3f}")


def test_ratios_dans_bornes():
    h1 = hist_simple(15, "V", "V", c2h=5)
    h2 = hist_simple(15, "D", "D", c2h=2)
    ratios, stats = apprendre_ratios_2h(h1, h2)
    for s in ("V", "N", "D"):
        assert RATIO_MIN <= ratios[s] <= RATIO_MAX
    print(f"  OK : ratios bornés {ratios}")


def test_ratios_shrinkage():
    """Avec peu de données, les ratios restent proches de 1.0."""
    # 3 matchs menés, 3 nuls, 3 gagnés, tous à 5 corners = ratio = 1
    h = hist_simple(15, "V", "V", c2h=4)  # tous V/V, 4 corners
    ratios, stats = apprendre_ratios_2h(h, h)
    # Le ratio V devrait être proche de 1.0 (tous les matchs sont V)
    assert abs(ratios["V"] - 1.0) < 0.5
    print(f"  OK : shrinkage vers 1.0, r_V={ratios['V']:.3f}")


def test_facteur_dans_bornes():
    ratios = {"V": 0.8, "N": 1.0, "D": 1.3}
    f = facteur_effet_score(0.3, 0.4, 0.3, ratios)
    assert FACTEUR_MIN <= f <= FACTEUR_MAX
    print(f"  OK : facteur borné {f:.3f}")


def test_facteur_penche_vers_trails():
    """Une équipe qui est souvent menée doit avoir un facteur > 1."""
    ratios = {"V": 0.8, "N": 1.0, "D": 1.4}
    f_menee = facteur_effet_score(0.1, 0.2, 0.7, ratios)
    f_lead = facteur_effet_score(0.7, 0.2, 0.1, ratios)
    assert f_menee > f_lead
    print(f"  OK : menée={f_menee:.3f} > mene={f_lead:.3f}")


def test_pipeline_complet():
    h1 = hist_simple(15, "V", "V")
    h2 = hist_simple(15, "D", "D")
    odds = {"home": 1.60, "draw": 3.80, "away": 4.50}
    r = calculer_facteurs_match(h1, h2, odds)
    assert "facteur_home_2h" in r
    assert "facteur_away_2h" in r
    assert FACTEUR_MIN <= r["facteur_home_2h"] <= FACTEUR_MAX
    assert FACTEUR_MIN <= r["facteur_away_2h"] <= FACTEUR_MAX
    print(f"  OK : facteurs home={r['facteur_home_2h']:.3f}, "
          f"away={r['facteur_away_2h']:.3f}")


def test_favori_a_facteur_plus_bas():
    """Avec des cotes normales et des données neutres, le favori a un
    facteur plus bas que l'outsider (car il mène plus souvent au HT)."""
    h = hist_simple(15, "N", "N")  # neutre : même nombre dans chaque scénario
    odds = {"home": 1.30, "draw": 5.0, "away": 8.0}  # home grand favori
    r = calculer_facteurs_match(h, h, odds)
    # Le home (favori) doit avoir un facteur ≤ celui de l'away
    assert r["facteur_home_2h"] <= r["facteur_away_2h"] + 1e-9
    print(f"  OK : favori={r['facteur_home_2h']:.3f} ≤ outsider={r['facteur_away_2h']:.3f}")


if __name__ == "__main__":
    tests = [
        test_retirer_marge_somme_a_un,
        test_retirer_marge_equilibre,
        test_transition_somme_ligne_un,
        test_transition_smoothed,
        test_proba_ht_equipe,
        test_ratios_dans_bornes,
        test_ratios_shrinkage,
        test_facteur_dans_bornes,
        test_facteur_penche_vers_trails,
        test_pipeline_complet,
        test_favori_a_facteur_plus_bas,
    ]
    print("=== Tests effet score ===\n")
    r = 0
    for t in tests:
        try:
            t(); r += 1
        except Exception as e:
            print(f"  ECHEC : {t.__name__} -> {e}")
    print(f"\n=== {r}/{len(tests)} tests reussis ===")
    if r != len(tests):
        raise SystemExit(1)
