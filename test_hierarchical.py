"""Tests du modele hierarchique."""
import numpy as np
from core.hierarchical import (
    estimate_alpha_beta, posterior_alpha_beta, nb_pmf,
    build_pmf_equipe, build_joint, build_mc_global,
)
from core.product_law import product_pmf, convolve


def hist_avec_moyennes(a1, a2):
    """Construit un historique bidon a partir de listes A1, A2."""
    return [
        {"date": f"2026-01-{i+1:02d}", "opponent": "X", "venue": "H",
         "corners_1h": a1[i], "corners_2h": a2[i]}
        for i in range(len(a1))
    ]


def test_alpha_beta_coherents():
    v = [2, 3, 4, 5, 3, 2, 4, 3, 5, 2]
    a, b = estimate_alpha_beta(v)
    assert a > 0 and b > 0
    mu = np.mean(v)
    mu_est = a / b
    # mu doit etre raisonnablement proche de la moyenne observee
    assert abs(mu_est - mu) / mu < 0.5, f"mu_est={mu_est} vs mu={mu}"
    print(f"  OK : alpha={a:.3f} beta={b:.3f} mu_est={mu_est:.3f} mu_obs={mu:.3f}")


def test_posterior_conjugue():
    a0, b0 = 5.0, 2.0
    v = [3, 4, 5]
    a1, b1 = posterior_alpha_beta(v, a0, b0)
    assert abs(a1 - (a0 + sum(v))) < 1e-9
    assert abs(b1 - (b0 + len(v))) < 1e-9
    print("  OK : posterior conjugue correct")


def test_nb_pmf_somme_1():
    pmf = nb_pmf(30, alpha=5.0, beta=2.0)
    assert abs(pmf.sum() - 1.0) < 1e-9
    assert all(p >= 0 for p in pmf)
    print("  OK : NB pmf somme a 1")


def test_shrinkage_vers_ligue():
    """Une equipe avec une moyenne extreme doit etre tiree vers la moyenne."""
    # Serie ou moyenne = 5, avec variance faible
    h = hist_avec_moyennes([5]*15, [5]*15)
    r = build_pmf_equipe(h)
    mu_a1 = np.sum(np.arange(len(r["A1"])) * r["A1"])
    # Avec shrinkage, mu_a1 doit rester proche de 5 mais pas exploser
    assert 4.0 < mu_a1 < 6.0
    print(f"  OK : shrinkage coherence, mu_a1={mu_a1:.3f}")


def test_build_joint_dimensions():
    p1 = nb_pmf(20, 5.0, 2.0)
    p2 = nb_pmf(25, 5.0, 2.0)
    j = build_joint(p1, p2)
    assert j.shape == (21, 26)
    assert abs(j.sum() - 1.0) < 1e-9
    print("  OK : joint aux bonnes dimensions et somme a 1")


def test_pipeline_complet():
    h_home = hist_avec_moyennes([3]*15, [4]*15)
    h_away = hist_avec_moyennes([2]*15, [3]*15)
    r = build_mc_global(h_home, h_away)
    for cle in ("mc1_pmf", "mc2_pmf", "global_pmf"):
        assert cle in r
        s = sum(r[cle].values())
        assert abs(s - 1.0) < 1e-9, f"{cle} somme={s}"
    print("  OK : pipeline complet, toutes les pmf somment a 1")


def test_global_moyenne_coherente():
    """Moyenne de GLOBAL = Moyenne(MC1) + Moyenne(MC2)."""
    h_home = hist_avec_moyennes([3]*15, [4]*15)
    h_away = hist_avec_moyennes([2]*15, [3]*15)
    r = build_mc_global(h_home, h_away)

    def moy(pmf):
        return sum(k * v for k, v in pmf.items())

    m1 = moy(r["mc1_pmf"])
    m2 = moy(r["mc2_pmf"])
    mg = moy(r["global_pmf"])
    assert abs(mg - (m1 + m2)) < 1e-9, f"mg={mg}, m1+m2={m1+m2}"
    print(f"  OK : mg={mg:.3f} = m1({m1:.3f}) + m2({m2:.3f})")


if __name__ == "__main__":
    tests = [
        test_alpha_beta_coherents,
        test_posterior_conjugue,
        test_nb_pmf_somme_1,
        test_shrinkage_vers_ligue,
        test_build_joint_dimensions,
        test_pipeline_complet,
        test_global_moyenne_coherente,
    ]
    print("=== Tests du modele hierarchique ===\n")
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
