"""Tests de la copule de Frank + dependance inter-equipes."""
import numpy as np
from core.copula import (
    pmf_joint_frank, tau_from_theta, theta_from_tau,
    estimer_theta, joint_avec_copule,
    rho_vers_tau_gaussien,
    estimer_correlation_inter_equipes,
    joint_mc_via_copule, global_depuis_joint_mc,
    RHO_SEUIL,
)


def test_independance_theta_zero():
    px = np.array([0.2, 0.3, 0.3, 0.15, 0.05])
    py = np.array([0.1, 0.25, 0.3, 0.2, 0.15])
    joint = pmf_joint_frank(px, py, 0.0)
    attendu = np.outer(px, py)
    assert np.allclose(joint, attendu, atol=1e-8)
    print("  OK : theta=0 -> independance")


def test_somme_a_un():
    px = np.array([0.2, 0.3, 0.3, 0.15, 0.05])
    py = np.array([0.1, 0.25, 0.3, 0.2, 0.15])
    for theta in (-5.0, -1.0, 0.0, 1.0, 5.0):
        joint = pmf_joint_frank(px, py, theta)
        assert abs(joint.sum() - 1.0) < 1e-8
    print("  OK : joint somme a 1")


def test_dependance_positive():
    px = np.array([0.2, 0.3, 0.3, 0.15, 0.05])
    py = np.array([0.1, 0.25, 0.3, 0.2, 0.15])
    j_ind = pmf_joint_frank(px, py, 0.0)
    j_dep = pmf_joint_frank(px, py, 5.0)
    diag_ind = sum(j_ind[k, k] for k in range(5))
    diag_dep = sum(j_dep[k, k] for k in range(5))
    assert diag_dep > diag_ind
    print(f"  OK : diagonale {diag_ind:.3f} -> {diag_dep:.3f}")


def test_tau_theta_monotone():
    thetas = [-10, -5, -1, 0, 1, 5, 10]
    taus = [tau_from_theta(t) for t in thetas]
    for i in range(len(taus) - 1):
        assert taus[i] < taus[i+1]
    print("  OK : tau monotone")


def test_theta_tau_inverse():
    for theta in (-5.0, -2.0, -0.5, 0.5, 2.0, 5.0):
        tau = tau_from_theta(theta)
        theta_back = theta_from_tau(tau)
        assert abs(theta_back - theta) < 0.01
    print("  OK : inversion tau <-> theta")


def test_estimation_donnees_correlees():
    rng = np.random.default_rng(42)
    n = 100
    z = rng.normal(0, 1, n)
    x = np.maximum(0, np.round(3 + 1.5 * z + rng.normal(0, 0.5, n))).astype(int)
    y = np.maximum(0, np.round(3 + 1.5 * z + rng.normal(0, 0.5, n))).astype(int)
    theta, tau, indep = estimer_theta(x, y)
    assert theta > 0 and not indep
    print(f"  OK : theta estime = {theta:.3f}")


def test_estimation_independance():
    rng = np.random.default_rng(7)
    n = 100
    x = rng.poisson(3, n)
    y = rng.poisson(3, n)
    theta, tau, indep = estimer_theta(x, y)
    assert indep
    print(f"  OK : independance detectee")


def test_rho_vers_tau():
    assert abs(rho_vers_tau_gaussien(0.0)) < 1e-9
    assert rho_vers_tau_gaussien(0.5) > 0
    assert rho_vers_tau_gaussien(-0.5) < 0
    print("  OK : rho -> tau coherent")


def hist_synthetique(rho_voulu=0.0, seed=1, n=30):
    """Genere un historique ou corners et xg_contre sont correles."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, n)
    bruit1 = rng.normal(0, 1, n)
    bruit2 = rng.normal(0, 1, n)
    xg_contre = 1.0 + 0.5 * z + 0.3 * bruit2
    c_tot = 6 + rho_voulu * 2 * z + (1 - rho_voulu) * 1.5 * bruit1
    c_tot = np.maximum(0, np.round(c_tot)).astype(int)
    xg_contre = np.maximum(0.1, xg_contre)
    hist = []
    for i in range(n):
        c1 = int(c_tot[i] // 2)
        c2 = int(c_tot[i] - c1)
        hist.append({
            "date": f"2026-01-{(i % 28) + 1:02d}", "opponent": f"T{i}", "venue": "H",
            "corners_1h": c1, "corners_2h": c2,
            "resultat_final": "V", "resultat_mt": "N",
            "cartons_rouges": 0,
            "xg_pour": 1.5, "xg_contre": float(xg_contre[i]),
            "xgot_pour": 1.2, "xgot_contre": 0.7,
        })
    return hist


def test_estimation_correlation_nulle():
    """
    Des donnees sans correlation -> rho doit rester petit.
    Avec n=60 (30+30), le bruit d'echantillonnage d'un rho nul a un
    ecart-type d'environ 1/sqrt(60) ~ 0.13. On tolere donc jusqu'a 0.45
    (environ 3.5 sigma) pour eviter les faux positifs de test.
    """
    h1 = hist_synthetique(rho_voulu=0.0, seed=1)
    h2 = hist_synthetique(rho_voulu=0.0, seed=2)
    rho, n, methode = estimer_correlation_inter_equipes(h1, h2)
    assert abs(rho) < 0.45, f"rho devrait etre petit, obtenu {rho}"
    print(f"  OK : correlation faible, rho={rho:.3f}")


def test_estimation_correlation_forte():
    h1 = hist_synthetique(rho_voulu=0.9, seed=3)
    h2 = hist_synthetique(rho_voulu=0.9, seed=4)
    rho, n, methode = estimer_correlation_inter_equipes(h1, h2)
    assert rho > 0.3, f"rho devrait etre > 0.3, obtenu {rho}"
    print(f"  OK : correlation forte detectee, rho={rho:.3f}")


def test_joint_mc_via_copule_independance():
    mc1 = np.array([0.3, 0.4, 0.3])
    mc2 = np.array([0.2, 0.5, 0.3])
    joint, info = joint_mc_via_copule(mc1, mc2, rho=0.0)
    assert info["independance_ok"]
    assert abs(joint.sum() - 1.0) < 1e-9
    print("  OK : rho=0 -> independance retenue")


def test_joint_mc_via_copule_dependance():
    mc1 = np.array([0.2, 0.3, 0.3, 0.2])
    mc2 = np.array([0.2, 0.3, 0.3, 0.2])
    joint, info = joint_mc_via_copule(mc1, mc2, rho=0.6)
    assert not info["independance_ok"]
    assert info["theta"] != 0
    assert abs(joint.sum() - 1.0) < 1e-9
    print(f"  OK : rho=0.6 -> theta={info['theta']:.3f}")


def test_global_depuis_joint():
    mc1 = np.array([0.2, 0.3, 0.3, 0.2])
    mc2 = np.array([0.3, 0.4, 0.3])
    joint, _ = joint_mc_via_copule(mc1, mc2, rho=0.3)
    g = global_depuis_joint_mc(joint)
    m1 = sum(k * p for k, p in enumerate(mc1))
    m2 = sum(k * p for k, p in enumerate(mc2))
    mg = sum(k * p for k, p in g.items())
    assert abs(mg - (m1 + m2)) < 1e-9
    print(f"  OK : mg={mg:.3f} = m1({m1:.3f}) + m2({m2:.3f})")


if __name__ == "__main__":
    tests = [
        test_independance_theta_zero, test_somme_a_un,
        test_dependance_positive, test_tau_theta_monotone,
        test_theta_tau_inverse, test_estimation_donnees_correlees,
        test_estimation_independance,
        test_rho_vers_tau,
        test_estimation_correlation_nulle,
        test_estimation_correlation_forte,
        test_joint_mc_via_copule_independance,
        test_joint_mc_via_copule_dependance,
        test_global_depuis_joint,
    ]
    print("=== Tests copule + dependance inter-equipes ===\n")
    r = 0
    for t in tests:
        try:
            t(); r += 1
        except Exception as e:
            print(f"  ECHEC : {t.__name__} -> {e}")
    print(f"\n=== {r}/{len(tests)} tests reussis ===")
    if r != len(tests):
        raise SystemExit(1)
