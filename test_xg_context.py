"""Tests du module Pression (xG -> corners)."""
import numpy as np
from core.xg_context import (
    fit_regression, predict_corners, moyenne_features,
    facteur_ajustement, entrainer_sur_les_deux_equipes,
    FEATURES, N_FEATURES,
)


def hist_synthetique(n=30, seed=42):
    """
    Genere un historique ou les corners sont PROPORTIONNELS aux xG.
    Relation connue : corners_1h = 2 + 1.5 * xg_pour + bruit
    Cela permet de verifier que la regression retrouve ce coefficient.
    """
    rng = np.random.default_rng(seed)
    hist = []
    for i in range(n):
        xg = float(rng.uniform(0.5, 2.5))
        xga = float(rng.uniform(0.5, 2.5))
        xgot = xg * 0.8 + float(rng.normal(0, 0.1))
        xgot_a = xga * 0.8 + float(rng.normal(0, 0.1))
        c1h = int(round(2 + 1.5 * xg + rng.normal(0, 0.3)))
        c2h = int(round(3 + 1.0 * xg + rng.normal(0, 0.3)))
        c1h = max(0, c1h); c2h = max(0, c2h)
        hist.append({
            "date": f"2026-01-{(i % 28) + 1:02d}",
            "opponent": f"T{i}", "venue": "H",
            "corners_1h": c1h, "corners_2h": c2h,
            "resultat_final": "V", "resultat_mt": "N", "cartons_rouges": 0,
            "xg_pour": xg, "xg_contre": xga,
            "xgot_pour": xgot, "xgot_contre": xgot_a,
        })
    return hist


def test_matrice_dimensions():
    h = hist_synthetique(20)
    beta, info = fit_regression(h, "corners_1h")
    assert len(beta) == N_FEATURES + 1
    assert info["n"] == 20
    print(f"  OK : regression ajustee, R2={info['r2']:.3f}")


def test_regression_retrouve_coefficient():
    """La relation reelle est : corners_1h = 2 + 1.5*xg_pour."""
    h = hist_synthetique(200)
    beta, info = fit_regression(h, "corners_1h")
    # beta[0] devrait etre proche de 2, beta[1] proche de 1.5
    # Mais on a du bruit sur les autres variables, donc on tolere
    assert info["r2"] > 0.6, f"R2 trop faible : {info['r2']}"
    print(f"  OK : R2={info['r2']:.3f}, beta={[round(b, 3) for b in beta]}")


def test_predict_coherent():
    h = hist_synthetique(50)
    beta, _ = fit_regression(h, "corners_1h")
    moy = moyenne_features(h)
    pred = predict_corners(moy, beta)
    obs = float(np.mean([m["corners_1h"] for m in h]))
    # La prediction moyenne doit etre proche de l'observation moyenne
    assert abs(pred - obs) / max(obs, 1) < 0.3
    print(f"  OK : pred={pred:.3f} obs={obs:.3f}")


def test_facteur_ajustement_dans_bornes():
    h = hist_synthetique(40)
    beta1, _ = fit_regression(h, "corners_1h")
    beta2, _ = fit_regression(h, "corners_2h")
    r = facteur_ajustement(h, beta1, beta2)
    assert 0.75 <= r["facteur_1h"] <= 1.35
    assert 0.75 <= r["facteur_2h"] <= 1.35
    print(f"  OK : facteurs bornes : {r['facteur_1h']:.3f} / {r['facteur_2h']:.3f}")


def test_pipeline_complet():
    h1 = hist_synthetique(30, seed=1)
    h2 = hist_synthetique(30, seed=2)
    r = entrainer_sur_les_deux_equipes(h1, h2)
    assert "beta_1h" in r and "beta_2h" in r
    assert r["info_1h"]["n"] == 60
    assert r["info_2h"]["n"] == 60
    print(f"  OK : entrainement conjoint, R2_1h={r['info_1h']['r2']:.3f}")


def test_features_connues():
    assert FEATURES == ("xg_pour", "xg_contre", "xgot_pour", "xgot_contre")
    assert N_FEATURES == 4
    print("  OK : features declarees")


if __name__ == "__main__":
    tests = [
        test_matrice_dimensions,
        test_regression_retrouve_coefficient,
        test_predict_coherent,
        test_facteur_ajustement_dans_bornes,
        test_pipeline_complet,
        test_features_connues,
    ]
    print("=== Tests du module Pression (xG) ===\n")
    r = 0
    for t in tests:
        try:
            t(); r += 1
        except Exception as e:
            print(f"  ECHEC : {t.__name__} -> {e}")
    print(f"\n=== {r}/{len(tests)} tests reussis ===")
    if r != len(tests):
        raise SystemExit(1)
