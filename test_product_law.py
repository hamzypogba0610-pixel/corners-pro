"""
Tests de verification du module product_law.
Si tout est vert : le coeur mathematique du bot fonctionne.
"""
import numpy as np
from core.product_law import (
    product_pmf, convolve, p_over, p_under,
    variance_over_mean, margin_removal_proportional, margin_removal_power,
)


def test_pmf_somme_a_un():
    px = np.array([0.2, 0.3, 0.3, 0.15, 0.05])
    py = np.array([0.1, 0.25, 0.3, 0.2, 0.15])
    joint = np.outer(px, py)
    pmf = product_pmf(joint)
    total = sum(pmf.values())
    assert abs(total - 1.0) < 1e-9, f"La somme vaut {total} au lieu de 1"
    print("  OK : la loi somme bien a 1")


def test_regle_du_zero():
    px = np.array([0.3, 0.7])
    py = np.array([0.4, 0.6])
    joint = np.outer(px, py)
    pmf = product_pmf(joint)
    attendu = px[0] + py[0] - px[0] * py[0]
    assert abs(pmf[0] - attendu) < 1e-9
    print("  OK : regle du zero correcte")


def test_over_half_line():
    pmf = {12: 0.4, 13: 0.35, 14: 0.25}
    assert abs(p_over(pmf, 12.5) - 0.60) < 1e-9
    assert abs(p_under(pmf, 12.5) - 0.40) < 1e-9
    print("  OK : Over/Under 12.5 correct")


def test_ligne_entiere_refusee():
    pmf = {12: 0.5, 13: 0.5}
    try:
        p_over(pmf, 12.0)
        raise AssertionError("Devrait refuser une ligne entiere")
    except ValueError:
        print("  OK : ligne entiere bien refusee")


def test_convolution():
    mc1 = {0: 0.3, 4: 0.5, 9: 0.2}
    mc2 = {0: 0.4, 6: 0.4, 12: 0.2}
    g = convolve(mc1, mc2)
    m1 = sum(k * v for k, v in mc1.items())
    m2 = sum(k * v for k, v in mc2.items())
    mg = sum(k * v for k, v in g.items())
    assert abs(mg - (m1 + m2)) < 1e-9
    print("  OK : convolution correcte")


def test_poisson_ratio_proche_de_un():
    from scipy.stats import poisson
    lam = 3.0
    px = np.array([poisson.pmf(k, lam) for k in range(20)])
    joint = np.outer(px, px)
    r = variance_over_mean(joint, axis=0)
    assert abs(r - 1.0) < 0.1, f"Ratio {r} au lieu de ~1"
    print("  OK : Poisson donne bien ratio ~1")


def test_marges_coherentes():
    po, pu, m1 = margin_removal_proportional(1.90, 1.90)
    po2, pu2, m2 = margin_removal_power(1.90, 1.90)
    assert abs(po + pu - 1.0) < 1e-9
    assert abs(po2 + pu2 - 1.0) < 1e-9
    assert abs(po - po2) < 0.01
    print("  OK : les deux methodes de marge sont coherentes")


if __name__ == "__main__":
    tests = [
        test_pmf_somme_a_un,
        test_regle_du_zero,
        test_over_half_line,
        test_ligne_entiere_refusee,
        test_convolution,
        test_poisson_ratio_proche_de_un,
        test_marges_coherentes,
    ]
    print("=== Lancement des tests ===\n")
    reussis = 0
    for t in tests:
        try:
            t()
            reussis += 1
        except Exception as e:
            print(f"  ECHEC : {t.__name__} -> {e}")
    print(f"\n=== {reussis}/{len(tests)} tests reussis ===")
    if reussis != len(tests):
        raise SystemExit(1)
