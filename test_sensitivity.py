"""Tests du module sensibilite (tornade)."""
import numpy as np
from core.sensitivity import perturber_hist, SCENARIOS


def hist_bidon():
    return [
        {"date": f"2026-01-{i+1:02d}", "opponent": f"X{i}", "venue": "H",
         "corners_1h": 3, "corners_2h": 4,
         "resultat_final": "V", "resultat_mt": "N", "cartons_rouges": 0,
         "xg_pour": 1.5, "xg_contre": 0.9,
         "xgot_pour": 1.2, "xgot_contre": 0.7}
        for i in range(15)
    ]


def test_scenarios_definis():
    noms = [s[0] for s in SCENARIOS]
    for attendu in ("moyenne_basse", "moyenne_haute", "sans_extremes"):
        assert attendu in noms, f"scenario manquant : {attendu}"
    assert len(SCENARIOS) == 3, f"attendu 3 scenarios, obtenu {len(SCENARIOS)}"
    print(f"  OK : 3 scenarios definis (tornade reduite)")


def test_perturber_ne_modifie_pas_original():
    h = hist_bidon()
    h_copy = [dict(m) for m in h]
    _ = perturber_hist(h, facteur_moyenne=0.5)
    for i in range(15):
        assert h[i] == h_copy[i], "L'original a ete modifie"
    print("  OK : original preserve")


def test_facteur_moyenne_reduit():
    h = hist_bidon()
    h_bas = perturber_hist(h, facteur_moyenne=0.5)
    c1 = sum(m["corners_1h"] for m in h_bas)
    assert c1 < sum(m["corners_1h"] for m in h)
    print(f"  OK : 0.5x -> {c1} corners 1h cumules (baseline 45)")


def test_facteur_moyenne_augmente():
    h = hist_bidon()
    h_haut = perturber_hist(h, facteur_moyenne=1.5)
    c1 = sum(m["corners_1h"] for m in h_haut)
    assert c1 > sum(m["corners_1h"] for m in h)
    print(f"  OK : 1.5x -> {c1} corners 1h cumules")


def test_retirer_extremes_garde_n():
    h = hist_bidon()
    h[0]["corners_1h"] = 0
    h[0]["corners_2h"] = 0
    h[1]["corners_1h"] = 15
    h[1]["corners_2h"] = 15
    h_p = perturber_hist(h, retirer_extremes=2)
    assert len(h_p) == 15, f"n doit rester 15, obtenu {len(h_p)}"
    print(f"  OK : n=15 apres retrait extremes")


def test_extremes_sont_moderes():
    """Les extremes doivent etre remplaces par la mediane."""
    h = hist_bidon()
    h[0]["corners_1h"] = 0
    h[0]["corners_2h"] = 0
    h[1]["corners_1h"] = 15
    h[1]["corners_2h"] = 15
    h_p = perturber_hist(h, retirer_extremes=2)
    vals_1h = sorted(m["corners_1h"] for m in h_p)
    assert vals_1h[0] > 0, "0 devrait avoir disparu"
    assert vals_1h[-1] < 15, "15 devrait avoir disparu"
    print(f"  OK : extremes moderes, min={vals_1h[0]}, max={vals_1h[-1]}")


def test_perturbation_neutre():
    """facteur_moyenne=1.0 -> copie identique."""
    h = hist_bidon()
    h_p = perturber_hist(h, facteur_moyenne=1.0)
    assert h_p == h
    print("  OK : 1.0x -> identique")


if __name__ == "__main__":
    tests = [
        test_scenarios_definis,
        test_perturber_ne_modifie_pas_original,
        test_facteur_moyenne_reduit,
        test_facteur_moyenne_augmente,
        test_retirer_extremes_garde_n,
        test_extremes_sont_moderes,
        test_perturbation_neutre,
    ]
    print("=== Tests sensibilite ===\n")
    r = 0
    for t in tests:
        try:
            t(); r += 1
        except Exception as e:
            print(f"  ECHEC : {t.__name__} -> {e}")
    print(f"\n=== {r}/{len(tests)} tests reussis ===")
    if r != len(tests):
        raise SystemExit(1)
