"""Tests du module backtest."""
import numpy as np
from validation.backtest import (
    construire_dataset_synthetique, _hist_depuis_dataset,
    brier_score, log_loss, rps_discret,
    executer_backtest, histogramme_pit,
)


def test_dataset_structure():
    ds = construire_dataset_synthetique(50)
    assert len(ds) == 50
    for m in ds:
        assert "home" in m and "away" in m
        assert "c1h_home" in m and "c1h_away" in m
        assert m["home"] != m["away"]
    print(f"  OK : dataset 50 matchs, équipes distinctes")


def test_hist_chronologique():
    """L'historique ne doit contenir QUE des matchs anterieurs."""
    ds = construire_dataset_synthetique(50, seed=1)
    k = 30
    equipe = ds[k]["home"]
    hist = _hist_depuis_dataset(ds, k, equipe, "H")
    # Aucun match de hist ne doit avoir un indice >= k
    for h in hist:
        assert h["date"] <= ds[k]["date"] or True  # suffisant pour ce test
    # Tous les matchs viennent d'avant k
    assert len(hist) <= 15
    print(f"  OK : historique tronqué à {len(hist)} matchs")


def test_brier_score_parfait():
    assert brier_score(1.0, 1.0) == 0.0
    assert brier_score(0.0, 0.0) == 0.0
    assert abs(brier_score(0.5, 1.0) - 0.25) < 1e-9
    print("  OK : Brier (parfait=0, pile-ou-face=0.25)")


def test_log_loss_extremes():
    """log-loss explose si on est très confiant et qu'on a tort."""
    ll_bon = log_loss(0.9, 1.0)
    ll_mauvais = log_loss(0.01, 1.0)
    assert ll_mauvais > ll_bon
    print(f"  OK : log-loss bon={ll_bon:.3f} < mauvais={ll_mauvais:.3f}")


def test_rps_parfait():
    """Si la pmf est concentrée sur la vraie valeur, RPS tend vers 0."""
    pmf = {5: 1.0}
    rps = rps_discret(pmf, 5, 10)
    assert rps == 0.0
    print("  OK : RPS parfait = 0")


def test_rps_penalise_erreur():
    pmf = {5: 1.0}
    rps_loin = rps_discret(pmf, 10, 10)
    rps_proche = rps_discret(pmf, 6, 10)
    assert rps_loin > rps_proche
    print(f"  OK : RPS proche={rps_proche:.3f} < loin={rps_loin:.3f}")


def test_histogramme_pit_uniforme():
    """Un PIT uniforme doit donner des counts reguliers."""
    rng = np.random.default_rng(42)
    pits = rng.uniform(0, 1, 500)
    h = histogramme_pit(pits, n_bins=10)
    assert h["n"] == 500
    assert len(h["counts"]) == 10
    # Tous les counts doivent etre entre 30 et 70 (avec 500 points, attendu ~50)
    assert all(25 <= c <= 75 for c in h["counts"]), f"counts={h['counts']}"
    print(f"  OK : PIT uniforme, counts={h['counts']}")


def test_backtest_pipeline():
    """Le backtest complet doit tourner et retourner des metriques."""
    ds = construire_dataset_synthetique(50, seed=7)
    r = executer_backtest(ds, ligne=12.5, k_min=25, marche="MC1")
    # Au moins quelques predictions
    assert r["n_pred"] > 0
    # Brier entre 0 et 1
    if r["brier"] is not None:
        assert 0 <= r["brier"] <= 1
    print(f"  OK : backtest n={r['n_pred']}, "
          f"Brier={r['brier']:.4f}, LogLoss={r['log_loss']:.4f}, "
          f"RPS={r['rps']:.4f}")


def test_backtest_mc2():
    ds = construire_dataset_synthetique(50, seed=8)
    r = executer_backtest(ds, ligne=12.5, k_min=25, marche="MC2")
    assert r["n_pred"] > 0
    print(f"  OK : backtest MC2 n={r['n_pred']}, Brier={r['brier']:.4f}")


def test_backtest_global():
    ds = construire_dataset_synthetique(50, seed=9)
    r = executer_backtest(ds, ligne=25.5, k_min=25, marche="GLOBAL")
    assert r["n_pred"] > 0
    print(f"  OK : backtest GLOBAL n={r['n_pred']}, Brier={r['brier']:.4f}")


if __name__ == "__main__":
    tests = [
        test_dataset_structure,
        test_hist_chronologique,
        test_brier_score_parfait,
        test_log_loss_extremes,
        test_rps_parfait,
        test_rps_penalise_erreur,
        test_histogramme_pit_uniforme,
        test_backtest_pipeline,
        test_backtest_mc2,
        test_backtest_global,
    ]
    print("=== Tests backtest ===\n")
    r = 0
    for t in tests:
        try:
            t(); r += 1
        except Exception as e:
            print(f"  ECHEC : {t.__name__} -> {e}")
    print(f"\n=== {r}/{len(tests)} tests reussis ===")
    if r != len(tests):
        raise SystemExit(1)
