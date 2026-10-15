"""Tests du module descriptif."""
from core.descriptive import (
    stats_brutes, frequence_zeros, surdispersion,
    correlation_pearson, correlation_spearman,
    calcul_mc, rapport_equipe, rapport_global,
)


def hist_bidon():
    return [
        {"date": "2026-01-01", "opponent": "A", "venue": "H",
         "corners_1h": 3, "corners_2h": 4},
        {"date": "2026-01-08", "opponent": "B", "venue": "H",
         "corners_1h": 2, "corners_2h": 3},
        {"date": "2026-01-15", "opponent": "C", "venue": "H",
         "corners_1h": 5, "corners_2h": 2},
        {"date": "2026-01-22", "opponent": "D", "venue": "H",
         "corners_1h": 0, "corners_2h": 4},
        {"date": "2026-01-29", "opponent": "E", "venue": "H",
         "corners_1h": 4, "corners_2h": 5},
    ]


def test_stats_brutes():
    s = stats_brutes([1, 2, 3, 4, 5])
    assert s["n"] == 5
    assert abs(s["moyenne"] - 3.0) < 1e-9
    assert abs(s["mediane"] - 3.0) < 1e-9
    assert s["min"] == 1 and s["max"] == 5
    print("  OK : stats de base")


def test_frequence_zeros():
    assert abs(frequence_zeros([0, 1, 0, 2, 0]) - 0.6) < 1e-9
    print("  OK : frequence zeros")


def test_surdispersion():
    v = [1, 2, 3, 4, 5]
    r = surdispersion(v)
    assert r > 0
    print(f"  OK : surdispersion = {r:.3f}")


def test_correlation():
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    assert abs(correlation_pearson(x, y) - 1.0) < 1e-9
    assert abs(correlation_spearman(x, y) - 1.0) < 1e-9
    y2 = [5, 4, 3, 2, 1]
    assert abs(correlation_pearson(x, y2) + 1.0) < 1e-9
    print("  OK : correlations")


def test_calcul_mc():
    h = hist_bidon()
    r = calcul_mc(h)
    assert r["A1"] == [3, 2, 5, 0, 4]
    assert r["MC"] == [12, 6, 10, 0, 20]
    print("  OK : calcul MC")


def test_rapport_equipe():
    r = rapport_equipe(hist_bidon(), label="Test")
    assert r["label"] == "Test"
    assert "A1" in r and "MC" in r
    assert r["MC"]["stats"]["n"] == 5
    assert r["MC"]["freq_zero"] == 0.2
    print("  OK : rapport equipe complet")


def test_rapport_global():
    h1 = hist_bidon()
    h2 = hist_bidon()
    r = rapport_global(h1, h2)
    assert "GLOBAL" in r
    assert r["MC1"]["stats"]["n"] == 5
    assert r["GLOBAL"]["stats"]["n"] == 5
    print("  OK : rapport global")


if __name__ == "__main__":
    tests = [
        test_stats_brutes,
        test_frequence_zeros,
        test_surdispersion,
        test_correlation,
        test_calcul_mc,
        test_rapport_equipe,
        test_rapport_global,
    ]
    print("=== Tests du descriptif ===\n")
    r = 0
    for t in tests:
        try:
            t()
            r += 1
        except Exception as e:
            print(f"  ECHEC : {t.__name__} -> {e}")
    print(f"\n=== {r}/{len(tests)} tests reussis ===")
    if r != len(tests):
        raise SystemExit(1)
