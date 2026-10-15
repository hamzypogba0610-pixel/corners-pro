"""Tests du module inflation de zeros."""
import numpy as np
from core.zero_inflation import (
    tester_inflation, pmf_zinb, _p_zero_nb,
    SEUIL_ECART, PI_MAX, PI_SEUIL_ACTIF,
)


def test_pas_inflation():
    """Sous Poisson, P(0) observe ~ P(0) attendu -> pas d'inflation."""
    rng = np.random.default_rng(42)
    v = rng.poisson(3.0, 100)
    # NB quasi-Poisson
    alpha, beta = 100.0, 100.0 / 3.0
    info = tester_inflation(v, alpha, beta)
    assert not info["inflation"]
    print(f"  OK : pas d'inflation (ecart={info['ecart']:.3f})")


def test_inflation_detectee():
    """Beaucoup de zeros -> inflation detectee."""
    v = np.array([0]*12 + [3, 4, 5, 2, 3, 4, 5, 3])
    alpha, beta = 5.0, 1.5  # NB classique
    info = tester_inflation(v, alpha, beta)
    assert info["inflation"], f"devrait detecter, ecart={info['ecart']}"
    assert 0 < info["pi"] <= PI_MAX
    print(f"  OK : inflation detectee, pi={info['pi']:.3f}")


def test_shrinkage_petit_n():
    """Avec n=15, pi doit etre bien plus petit que pi_brut."""
    v = np.array([0]*12 + [3, 4, 5])
    alpha, beta = 5.0, 1.5
    info = tester_inflation(v, alpha, beta)
    if info["inflation"]:
        assert info["pi"] < info["pi_brut"]
        w = info["n"] / (info["n"] + 5.0)
        assert abs(info["pi"] - info["pi_brut"] * w) < 1e-9
    print(f"  OK : shrinkage n=15, pi={info['pi']:.3f} < brut={info['pi_brut']:.3f}")


def test_pi_borne():
    """pi ne doit jamais depasser PI_MAX."""
    v = np.array([0]*100)
    alpha, beta = 5.0, 1.5
    info = tester_inflation(v, alpha, beta)
    assert info["pi_brut"] <= PI_MAX
    print(f"  OK : pi borne a {PI_MAX}")


def test_zinb_somme_a_un():
    pmf = pmf_zinb(30, alpha=5.0, beta=2.0, pi=0.2)
    assert abs(pmf.sum() - 1.0) < 1e-9
    assert all(p >= 0 for p in pmf)
    print("  OK : ZINB somme a 1")


def test_zinb_meme_alpha_beta_plus_de_zero():
    """A alpha/beta fixes, ZINB met plus de masse sur 0 que NB."""
    from scipy.stats import nbinom
    k_max = 30
    alpha, beta = 5.0, 2.0
    p = beta / (beta + 1.0)
    pmf_nb = nbinom.pmf(np.arange(k_max + 1), alpha, p)
    pmf_nb = pmf_nb / pmf_nb.sum()
    pmf_zi = pmf_zinb(k_max, alpha, beta, 0.3)
    assert pmf_zi[0] > pmf_nb[0]
    print(f"  OK : P(0) NB={pmf_nb[0]:.3f} -> ZINB={pmf_zi[0]:.3f}")


def test_zinb_pi_zero_egale_nb():
    """ZINB avec pi=0 doit egaler NB exactement."""
    from scipy.stats import nbinom
    k_max = 30
    alpha, beta = 5.0, 2.0
    p = beta / (beta + 1.0)
    pmf_nb = nbinom.pmf(np.arange(k_max + 1), alpha, p)
    pmf_nb = pmf_nb / pmf_nb.sum()
    pmf_zi = pmf_zinb(k_max, alpha, beta, 0.0)
    assert np.allclose(pmf_zi, pmf_nb, atol=1e-9)
    print("  OK : ZINB(pi=0) == NB")


def test_p_zero_nb_coherent():
    """_p_zero_nb doit donner la bonne valeur analytique."""
    from scipy.stats import nbinom
    alpha, beta = 5.0, 2.0
    p = beta / (beta + 1.0)
    attendu = float(nbinom.pmf(0, alpha, p))
    assert abs(_p_zero_nb(alpha, beta) - attendu) < 1e-9
    print(f"  OK : P(0) NB = {attendu:.4f}")


if __name__ == "__main__":
    tests = [
        test_pas_inflation,
        test_inflation_detectee,
        test_shrinkage_petit_n,
        test_pi_borne,
        test_zinb_somme_a_un,
        test_zinb_meme_alpha_beta_plus_de_zero,
        test_zinb_pi_zero_egale_nb,
        test_p_zero_nb_coherent,
    ]
    print("=== Tests inflation de zeros ===\n")
    r = 0
    for t in tests:
        try:
            t(); r += 1
        except Exception as e:
            print(f"  ECHEC : {t.__name__} -> {e}")
    print(f"\n=== {r}/{len(tests)} tests reussis ===")
    if r != len(tests):
        raise SystemExit(1)
