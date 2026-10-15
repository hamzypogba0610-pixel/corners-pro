"""Tests des modules de decision (EV + gate)."""
import numpy as np
from decision.ev import (
    ev_over, ev_under, kelly_fraction, kelly_prudent,
    ev_posterior, edge_vs_implicite, penalite_winner_curse,
    PENALITE_WINNER_CURSE, KELY_PLAFOND,
)
from decision.gate import evaluer_selection, detecter_faux_signaux, verdict_texte


def test_ev_over_simple():
    """EV = 0 quand p correspond exactement a la cote sans marge."""
    assert abs(ev_over(0.5, 2.0)) < 1e-9
    assert ev_over(0.6, 2.0) > 0
    assert ev_over(0.4, 2.0) < 0
    print("  OK : EV Over basique")


def test_ev_under():
    assert abs(ev_under(0.5, 2.0)) < 1e-9
    print("  OK : EV Under")


def test_kelly_zero_si_ev_negatif():
    assert kelly_fraction(0.4, 2.0) == 0.0
    print("  OK : Kelly nul si EV negatif")


def test_kelly_positif_si_edge():
    f = kelly_fraction(0.6, 2.0)
    assert 0 < f < 1
    assert abs(f - 0.2) < 1e-9
    print(f"  OK : Kelly = {f:.3f}")


def test_kelly_prudent_plafonne():
    f = kelly_prudent(0.9, 5.0)
    assert f <= KELY_PLAFOND
    print(f"  OK : Kelly prudent plafonne a {f:.3f}")


def test_ev_posterior_bornes():
    rng = np.random.default_rng(42)
    p_samples = rng.beta(20, 10, 5000)
    r = ev_posterior(p_samples, cote=1.8)
    assert "ev_mediane" in r and "ev_p20" in r and "p_ev_positif" in r
    assert 0.0 <= r["p_ev_positif"] <= 1.0
    assert r["ev_p20"] <= r["ev_mediane"]
    print(f"  OK : EV posterior, med={r['ev_mediane']:.4f}, p20={r['ev_p20']:.4f}")


def test_edge_simple():
    assert abs(edge_vs_implicite(0.55, 0.50) - 0.05) < 1e-9
    print("  OK : edge calcule")


def test_penalite_winner_curse():
    assert abs(penalite_winner_curse(0.55) - (0.55 - PENALITE_WINNER_CURSE)) < 1e-9
    print("  OK : penalite winner's curse")


def test_gate_refus_par_defaut():
    """Un cas vide ou faible doit etre refuse."""
    analyse = {
        "n": 10,
        "p_estime": 0.5,
        "p_implicite_sans_marge": 0.5,
        "ev_p20": -0.05,
        "contexte": {},
    }
    r = evaluer_selection(analyse)
    assert r["accepte"] is False
    assert len(r["raisons_refus"]) >= 3
    print(f"  OK : refus par defaut ({len(r['raisons_refus'])} raisons)")


def test_gate_accepte_si_tout_bon():
    """Un cas parfait doit passer."""
    analyse = {
        "n": 20,
        "p_estime": 0.60,
        "p_implicite_sans_marge": 0.52,
        "ev_p20": 0.02,
        "spread_modeles": 0.03,
        "stable_sensibilite": True,
        "contexte": {},
    }
    r = evaluer_selection(analyse)
    assert r["accepte"] is True
    assert "RETENUE" in verdict_texte(r)
    print("  OK : selection valide retenue")


def test_gate_refuse_si_penalite_fait_passer_sous_seuil():
    """Edge 4.5 pts AVANT penalite -> 2.5 APRES -> refus."""
    analyse = {
        "n": 20,
        "p_estime": 0.565,
        "p_implicite_sans_marge": 0.52,
        "ev_p20": 0.01,
        "contexte": {},
    }
    r = evaluer_selection(analyse)
    assert r["accepte"] is False
    print("  OK : penalite winner's curse protege")


def test_faux_signaux_detectes():
    """Tous les seuils doivent etre franchis pour declencher 7 alertes."""
    ctx = {
        "cartons_rouges_recents": 2,      # seuil >= 2
        "cv_mc": 2.5,                     # seuil > 2.0
        "freq_zero_mc": 0.6,              # seuil > 0.5
        "cote_min": 1.20,                 # seuil < 1.30
        "pagerank_homogene": False,       # doit etre False
        "changement_entraineur": True,    # doit etre True
        "rotation_probable": True,        # doit etre True
    }
    alertes = detecter_faux_signaux(ctx)
    assert len(alertes) == 7, f"attendu 7, obtenu {len(alertes)} : {alertes}"
    print(f"  OK : 7 faux signaux detectes")


def test_faux_signaux_bloque_acceptation():
    analyse = {
        "n": 20,
        "p_estime": 0.60,
        "p_implicite_sans_marge": 0.50,
        "ev_p20": 0.05,
        "contexte": {"cartons_rouges_recents": 3},
    }
    r = evaluer_selection(analyse)
    assert r["accepte"] is False
    assert any("faux" in x.lower() or "Carton" in x or "carton" in x
               for x in r["raisons_refus"])
    print("  OK : faux signaux bloquent l'acceptation")


if __name__ == "__main__":
    tests = [
        test_ev_over_simple, test_ev_under,
        test_kelly_zero_si_ev_negatif, test_kelly_positif_si_edge,
        test_kelly_prudent_plafonne, test_ev_posterior_bornes,
        test_edge_simple, test_penalite_winner_curse,
        test_gate_refus_par_defaut, test_gate_accepte_si_tout_bon,
        test_gate_refuse_si_penalite_fait_passer_sous_seuil,
        test_faux_signaux_detectes, test_faux_signaux_bloque_acceptation,
    ]
    print("=== Tests de la decision ===\n")
    r = 0
    for t in tests:
        try:
            t(); r += 1
        except Exception as e:
            print(f"  ECHEC : {t.__name__} -> {e}")
    print(f"\n=== {r}/{len(tests)} tests reussis ===")
    if r != len(tests):
        raise SystemExit(1)
