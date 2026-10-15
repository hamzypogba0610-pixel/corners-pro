"""Tests de la copule de Frank."""
import numpy as np
from core.copula import (
    pmf_joint_frank, tau_from_theta, theta_from_tau,
    estimer_theta, joint_avec_copule,
)


def test_independance_theta_zero():
    """A theta = 0, la copule doit redonner le produit des marginales."""
    px = np.array([0.2, 0.3, 0.3, 0.15, 0.05])
    py = np.array([0.1, 0.25, 0.3, 0.2, 0.15])
    joint = pmf_joint_frank(px, py, 0.0)
    attendu = np.outer(px, py)
    assert np.allclose(joint, attendu, atol=1e-8), "theta=0 devrait donner independance"
    print("  OK : theta=0 -> independance retrouvee")


def test_somme_a_un():
    px = np.array([0.2, 0.3, 0.3, 0.15, 0.05])
    py = np.array([0.1, 0.25, 0.3, 0.2, 0.15])
    for theta in (-5.0, -1.0, 0.0, 1.0, 5.0):
        joint = pmf_joint_frank(px, py, theta)
        s = joint.sum()
        assert abs(s - 1.0) < 1e-8, f"theta={theta} : somme={s}"
    print("  OK : joint somme a 1 pour plusieurs theta")


def test_dependance_positive_renforce_diagonale():
    """A theta > 0, les couples concordants doivent etre plus probables."""
    px = np.array([0.2, 0.3, 0.3, 0.15, 0.05])
    py = np.array([0.1, 0.25, 0.3, 0.2, 0.15])
    j_ind = pmf_joint_frank(px, py, 0.0)
    j_dep = pmf_joint_frank(px, py, 5.0)
    # Diagonale : somme P(X=Y=k)
    diag_ind = sum(j_ind[k, k] for k in range(5))
    diag_dep = sum(j_dep[k, k] for k in range(5))
    assert diag_dep > diag_ind, "theta>0 devrait renforcer la diagonale"
    print(f"  OK : diagonale {diag_ind:.3f} -> {diag_dep:.3f}")


def test_tau_theta_monotone():
    """tau(theta) doit etre croissant."""
    thetas = [-10, -5, -1, 0, 1, 5, 10]
    taus = [tau_from_theta(t) for t in thetas]
    for i in range(len(taus) - 1):
        assert taus[i] < taus[i+1], f"tau non monotone : {taus}"
    print(f"  OK : tau monotone : {[round(t, 3) for t in taus]}")


def test_theta_tau_inverse():
    """theta_from_tau(tau_from_theta(t)) doit redonner theta."""
    for theta in (-5.0, -2.0, -0.5, 0.5, 2.0, 5.0):
        tau = tau_from_theta(theta)
        theta_back = theta_from_tau(tau)
        assert abs(theta_back - theta) < 0.01, (
            f"theta={theta} -> tau={tau} -> theta_back={theta_back}"
        )
    print("  OK : inversion tau <-> theta coherente")


def test_estimation_donnees_correlees():
    """Genere des donnees correlees positivement, doit estimer theta > 0."""
    rng = np.random.default_rng(42)
    n = 100
    # Variable latente partagee : deux comptages correles
    z = rng.normal(0, 1, n)
    x = np.maximum(0, np.round(3 + 1.5 * z + rng.normal(0, 0.5, n))).astype(int)
    y = np.maximum(0, np.round(3 + 1.5 * z + rng.normal(0, 0.5, n))).astype(int)
    theta, tau, indep = estimer_theta(x, y)
    assert theta > 0, f"theta devrait etre > 0, obtenu {theta}"
    assert not indep
    print(f"  OK : theta estime = {theta:.3f} (tau = {tau:.3f})")


def test_estimation_independance():
    """Genere des donnees independantes, doit retourner independance_ok=True."""
    rng = np.random.default_rng(7)
    n = 100
    x = rng.poisson(3, n)
    y = rng.poisson(3, n)
    theta, tau, indep = estimer_theta(x, y)
    assert indep, f"Devrait detecter l'independance, tau={tau}"
    print(f"  OK : independance detectee (tau={tau:.3f})")


def test_pipeline_avec_copule():
    """Le pipeline doit retourner une joint valide + les infos de dependance."""
    px = np.array([0.2, 0.3, 0.3, 0.15, 0.05])
    py = np.array([0.1, 0.25, 0.3, 0.2, 0.15])
    rng = np.random.default_rng(1)
    x_obs = rng.poisson(3, 50)
    y_obs = rng.poisson(3, 50)
    joint, info = joint_avec_copule(px, py, x_obs, y_obs)
    assert abs(joint.sum() - 1.0) < 1e-8
    assert "theta" in info and "tau" in info and "independance_ok" in info
    print(f"  OK : pipeline complet, theta={info['theta']:.3f}")


if __name__ == "__main__":
    tests = [
        test_independance_theta_zero,
        test_somme_a_un,
        test_dependance_positive_renforce_diagonale,
        test_tau_theta_monotone,
        test_theta_tau_inverse,
        test_estimation_donnees_correlees,
        test_estimation_independance,
        test_pipeline_avec_copule,
    ]
    print("=== Tests de la copule de Frank ===\n")
    r = 0
    for t in tests:
        try:
            t(); r += 1
        except Exception as e:
            print(f"  ECHEC : {t.__name__} -> {e}")
    print(f"\n=== {r}/{len(tests)} tests reussis ===")
    if r != len(tests):
        raise SystemExit(1)
