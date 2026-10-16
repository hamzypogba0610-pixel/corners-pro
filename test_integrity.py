"""Tests de la couche d'integrite."""
import numpy as np
from core.integrity import (
    verifier_pmf, verifier_pmf_array, verifier_proba, verifier_borne,
    verifier_coherence_convolution,
    verifier_parametres_nb, verifier_parametres_zinb,
    verifier_parametres_copule, verifier_facteurs,
    verifier_rapport_complet,
    TOLERANCE_PMF,
)


def test_pmf_valide():
    pmf = {0: 0.2, 1: 0.3, 2: 0.5}
    assert verifier_pmf(pmf) == []
    print("  OK : pmf valide")


def test_pmf_somme_incorrecte():
    pmf = {0: 0.2, 1: 0.3, 2: 0.4}
    pbs = verifier_pmf(pmf)
    assert len(pbs) >= 1
    assert "somme" in pbs[0]
    print("  OK : somme incorrecte detectee")


def test_pmf_valeur_negative():
    pmf = {0: -0.1, 1: 1.1}
    pbs = verifier_pmf(pmf)
    assert any("negative" in x for x in pbs)
    print("  OK : valeur negative detectee")


def test_pmf_nan():
    pmf = {0: float("nan"), 1: 1.0}
    pbs = verifier_pmf(pmf)
    assert any("non finie" in x for x in pbs)
    print("  OK : NaN detecte")


def test_pmf_array_valide():
    a = np.array([0.1, 0.2, 0.3, 0.4])
    assert verifier_pmf_array(a) == []
    print("  OK : pmf array valide")


def test_pmf_array_negatif():
    a = np.array([-0.1, 0.5, 0.6])
    pbs = verifier_pmf_array(a)
    assert len(pbs) >= 1
    print("  OK : array negatif detecte")


def test_proba_valide():
    assert verifier_proba(0.5) == []
    assert verifier_proba(0.0) == []
    assert verifier_proba(1.0) == []
    print("  OK : proba valide")


def test_proba_hors_bornes():
    assert len(verifier_proba(1.5)) >= 1
    assert len(verifier_proba(-0.1)) >= 1
    print("  OK : proba hors [0,1] detectee")


def test_borne():
    assert verifier_borne(5.0, 0.0, 10.0, "x") == []
    pbs = verifier_borne(15.0, 0.0, 10.0, "x")
    assert len(pbs) == 1
    print("  OK : borne")


def test_coherence_convolution_ok():
    mc1 = {0: 0.5, 4: 0.5}
    mc2 = {0: 0.5, 6: 0.5}
    # E[MC1]=2, E[MC2]=3, donc E[GLOBAL]=5
    # Loi de GLOBAL : 0->0.25, 4->0.25, 6->0.25, 10->0.25
    global_pmf = {0: 0.25, 4: 0.25, 6: 0.25, 10: 0.25}
    assert verifier_coherence_convolution(mc1, mc2, global_pmf) == []
    print("  OK : coherence convolution")


def test_coherence_convolution_ko():
    mc1 = {0: 0.5, 4: 0.5}
    mc2 = {0: 0.5, 6: 0.5}
    # Loi incorrecte (valeur decalee)
    global_pmf = {0: 0.25, 4: 0.25, 6: 0.25, 20: 0.25}
    pbs = verifier_coherence_convolution(mc1, mc2, global_pmf)
    assert len(pbs) >= 1
    print("  OK : incoherence convolution detectee")


def test_parametres_nb():
    assert verifier_parametres_nb(5.0, 2.0) == []
    assert len(verifier_parametres_nb(-1.0, 2.0)) >= 1
    assert len(verifier_parametres_nb(5.0, 0.0)) >= 1
    print("  OK : parametres NB")


def test_parametres_zinb():
    assert verifier_parametres_zinb(5.0, 2.0, 0.3) == []
    assert len(verifier_parametres_zinb(5.0, 2.0, 0.8)) >= 1
    print("  OK : parametres ZINB")


def test_parametres_copule():
    assert verifier_parametres_copule(2.0, 0.3) == []
    assert len(verifier_parametres_copule(35.0, 0.0)) >= 1
    assert len(verifier_parametres_copule(0.0, 1.5)) >= 1
    print("  OK : parametres copule")


def test_facteurs():
    assert verifier_facteurs(1.0, 1.05, 0.95) == []
    pbs = verifier_facteurs(5.0, 1.0, 1.0)
    assert len(pbs) >= 1
    print("  OK : facteurs")


def test_rapport_complet_ok():
    rapport = {
        "stats": {"MC1": {}, "MC2": {}, "GLOBAL": {}},
        "marches": {
            "MC1": {
                "cotes": {
                    "over": {"p": 0.5, "cote": 1.9,
                             "p_impl_prop": 0.5, "kelly": 0.0},
                    "under": {"p": 0.5, "cote": 1.9,
                              "p_impl_prop": 0.5, "kelly": 0.0},
                },
            },
        },
    }
    r = verifier_rapport_complet(rapport)
    assert r["ok"] is True
    print("  OK : rapport complet valide")


def test_rapport_complet_ko():
    rapport = {
        "stats": {"MC1": {}, "MC2": {}},  # GLOBAL manquant
        "marches": {
            "MC1": {
                "cotes": {
                    "over": {"p": 1.5, "cote": 1.9,  # p hors bornes
                             "p_impl_prop": 0.5, "kelly": 0.0},
                    "under": {"p": 0.5, "cote": 1.9,
                              "p_impl_prop": 0.5, "kelly": 0.0},
                },
            },
        },
    }
    r = verifier_rapport_complet(rapport)
    assert r["ok"] is False
    assert len(r["problemes_bloquants"]) >= 2
    print(f"  OK : rapport KO, {len(r['problemes_bloquants'])} bloquants")


def test_rapport_non_dict():
    r = verifier_rapport_complet("pas un dict")
    assert r["ok"] is False
    print("  OK : rapport non-dict refuse")


if __name__ == "__main__":
    tests = [
        test_pmf_valide,
        test_pmf_somme_incorrecte,
        test_pmf_valeur_negative,
        test_pmf_nan,
        test_pmf_array_valide,
        test_pmf_array_negatif,
        test_proba_valide,
        test_proba_hors_bornes,
        test_borne,
        test_coherence_convolution_ok,
        test_coherence_convolution_ko,
        test_parametres_nb,
        test_parametres_zinb,
        test_parametres_copule,
        test_facteurs,
        test_rapport_complet_ok,
        test_rapport_complet_ko,
        test_rapport_non_dict,
    ]
    print("=== Tests integrite ===\n")
    r = 0
    for t in tests:
        try:
            t(); r += 1
        except Exception as e:
            print(f"  ECHEC : {t.__name__} -> {e}")
    print(f"\n=== {r}/{len(tests)} tests reussis ===")
    if r != len(tests):
        raise SystemExit(1)
