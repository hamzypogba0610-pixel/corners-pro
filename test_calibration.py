"""Tests de la calibration (Temperature, Platt, Beta, ensemble)."""
import numpy as np
from core.calibration import (
    logit, sigmoid,
    calibrer_temperature, fit_temperature,
    calibrer_platt, fit_platt,
    calibrer_beta, fit_beta,
    combiner_ensembles, entrainer_calibrateurs, appliquer,
)


def test_logit_sigmoid_inverse():
    for p in (0.1, 0.3, 0.5, 0.7, 0.9):
        z = logit(p)
        p_back = float(sigmoid(z))
        assert abs(p_back - p) < 1e-9
    print("  OK : logit/sigmoid inverses")


def test_sigmoid_stable():
    """Numeriquement stable pour de grandes valeurs."""
    assert abs(float(sigmoid(1000.0)) - 1.0) < 1e-9
    assert abs(float(sigmoid(-1000.0)) - 0.0) < 1e-9
    print("  OK : sigmoid stable aux extremes")


def test_temperature_identite():
    """T=1 -> identite."""
    for p in (0.2, 0.5, 0.8):
        assert abs(calibrer_temperature(p, 1.0) - p) < 1e-9
    print("  OK : temperature T=1 = identite")


def test_temperature_ecrase():
    """T>1 -> ecrase vers 0.5."""
    p = 0.9
    q = calibrer_temperature(p, 2.0)
    assert q < p
    assert q > 0.5
    print(f"  OK : T=2 ecrase 0.9 -> {q:.4f}")


def test_platt_identite():
    """a=1, b=0 -> identite."""
    for p in (0.2, 0.5, 0.8):
        assert abs(calibrer_platt(p, 1.0, 0.0) - p) < 1e-9
    print("  OK : Platt a=1, b=0 = identite")


def test_beta_identite():
    """a=b=1, c=0 -> identite."""
    for p in (0.2, 0.5, 0.8):
        assert abs(calibrer_beta(p, 1.0, 1.0, 0.0) - p) < 1e-9
    print("  OK : Beta a=b=1, c=0 = identite")


def test_ensemble_sans_params():
    """Sans parametres, l'ensemble retourne p inchange."""
    for p in (0.2, 0.5, 0.8):
        assert abs(combiner_ensembles(p, None) - p) < 1e-9
    print("  OK : ensemble sans params = identite")


def test_ensemble_params_identite():
    """Avec params identite, l'ensemble retourne p inchange."""
    params = {
        "T": 1.0, "a_platt": 1.0, "b_platt": 0.0,
        "a_beta": 1.0, "b_beta": 1.0, "c_beta": 0.0,
    }
    for p in (0.2, 0.5, 0.8):
        assert abs(combiner_ensembles(p, params) - p) < 1e-9
    print("  OK : ensemble avec params identite = identite")


def test_fit_temperature_sur_donnees():
    """Avec des predictions sur-confiantes, T doit s'eloigner de 1."""
    rng = np.random.default_rng(42)
    n = 200
    # Vraie proba 0.6, mais le modele predit 0.85 systematiquement
    p_pred = np.full(n, 0.85)
    y = (rng.uniform(0, 1, n) < 0.6).astype(float)
    T = fit_temperature(p_pred, y)
    # T doit etre > 1 pour ecraser vers 0.5
    assert T > 1.0, f"T attendu > 1, obtenu {T}"
    print(f"  OK : fit T={T:.3f} (attend > 1)")


def test_fit_platt_sur_donnees():
    rng = np.random.default_rng(7)
    n = 300
    p_pred = rng.uniform(0.1, 0.9, n)
    # Cible : probabilites plus douces
    y = (rng.uniform(0, 1, n) < (p_pred * 0.7 + 0.15)).astype(float)
    a, b = fit_platt(p_pred, y)
    assert abs(a) > 0
    print(f"  OK : fit Platt a={a:.3f}, b={b:.3f}")


def test_fit_beta_sur_donnees():
    rng = np.random.default_rng(11)
    n = 300
    p_pred = rng.uniform(0.1, 0.9, n)
    y = (rng.uniform(0, 1, n) < p_pred).astype(float)
    a, b, c = fit_beta(p_pred, y)
    assert abs(a) > 0 and abs(b) > 0
    print(f"  OK : fit Beta a={a:.3f}, b={b:.3f}, c={c:.3f}")


def test_entrainer_pipeline():
    rng = np.random.default_rng(1)
    n = 100
    p_pred = rng.uniform(0.1, 0.9, n)
    y = (rng.uniform(0, 1, n) < p_pred).astype(float)
    params = entrainer_calibrateurs(p_pred, y)
    assert "T" in params and "a_platt" in params and "a_beta" in params
    assert params["n"] == 100
    print(f"  OK : entrainement complet, n={params['n']}")


def test_appliquer_sans_params():
    p, info = appliquer(0.7, None)
    assert abs(p - 0.7) < 1e-9
    assert info["calibree"] is False
    print("  OK : appliquer sans params = identite")


def test_appliquer_avec_params():
    params = {"T": 2.0, "a_platt": 1.0, "b_platt": 0.0,
              "a_beta": 1.0, "b_beta": 1.0, "c_beta": 0.0}
    p, info = appliquer(0.9, params)
    assert info["calibree"] is True
    assert 0.5 < p < 0.9
    print(f"  OK : appliquer avec T=2, 0.9 -> {p:.4f}")


if __name__ == "__main__":
    tests = [
        test_logit_sigmoid_inverse,
        test_sigmoid_stable,
        test_temperature_identite,
        test_temperature_ecrase,
        test_platt_identite,
        test_beta_identite,
        test_ensemble_sans_params,
        test_ensemble_params_identite,
        test_fit_temperature_sur_donnees,
        test_fit_platt_sur_donnees,
        test_fit_beta_sur_donnees,
        test_entrainer_pipeline,
        test_appliquer_sans_params,
        test_appliquer_avec_params,
    ]
    print("=== Tests calibration ===\n")
    r = 0
    for t in tests:
        try:
            t(); r += 1
        except Exception as e:
            print(f"  ECHEC : {t.__name__} -> {e}")
    print(f"\n=== {r}/{len(tests)} tests reussis ===")
    if r != len(tests):
        raise SystemExit(1)
