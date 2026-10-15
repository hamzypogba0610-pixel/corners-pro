"""Tests des modules de decision (EV + gate + IC)."""
import numpy as np
from decision.ev import (
    ev_over, ev_under, kelly_fraction, kelly_prudent,
    ev_posterior, edge_vs_implicite, penalite_winner_curse,
    PENALITE_WINNER_CURSE, KELY_PLAFOND,
)
from decision.gate import (
    evaluer_selection, detecter_faux_signaux, verdict_texte,
    IC80_LARGEUR_MAX,
)


def test_ev_over_simple():
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
    assert r["ev_p20"] <= r["ev_mediane"]
    print(f"  OK : EV posterior, med={r['ev_mediane']:.4f}, p20={r['ev_p20']:.4f}")


def test_ic80_present_et_ordonne():
    """L'IC80 doit etre present et p10 < p90."""
    rng = np.random.default_rng(7)
    p_samples = rng.beta(20, 10, 5000)
    r = ev_posterior(p_samples, cote=1.8)
    assert "ic80_bas" in r and "ic80_haut" in r
    assert "ic80_largeur" in r
    assert r["ic80_bas"] < r["ic80_haut"]
    assert abs(r["ic80_largeur"] - (r["ic80_haut"] - r["ic80_bas"])) < 1e-9
    print(f"  OK : IC80 = [{r['ic80_bas']:.3f} - {r['ic80_haut']:.3f}] "
          f"(largeur {r['ic80_largeur']:.3f})")


def test_ic80_serre_pour_n_grand():
    """Un beta avec n=1000 doit donner un IC80 plus serre qu'avec n=20."""
    rng = np.random.default_rng(1)
    p_serre = rng.beta(200, 60, 5000)
    p_large = rng.beta(4, 1.2, 5000)
    r_serre = ev_posterior(p_serre, 1.8)
    r_large = ev_posterior(p_large, 1.8)
    assert r_serre["ic80_largeur"] < r_large["ic80_largeur"]
    print(f"  OK : IC80 serre {r_serre['ic80_largeur']:.3f} "
          f"< large {r_large['ic80_largeur']:.3f}")


def test_edge_simple():
    assert abs(edge_vs_implicite(0.55, 0.50) - 0.05) < 1e-9
    print("  OK : edge calcule")


def test_penalite_winner_curse():
    assert abs(penalite_winner_curse(0.55) - (0.55 - PENALITE_WINNER_CURSE)) < 1e-9
    print("  OK : penalite winner's curse")


def test_gate_refus_par_defaut():
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
    analyse = {
        "n": 20,
        "p_estime": 0.60,
        "p_implicite_sans_marge": 0.52,
        "ev_p20": 0.02,
        "spread_modeles": 0.03,
        "stable_sensibilite": True,
        "ic80_largeur": 0.10,
        "contexte": {},
    }
    r = evaluer_selection(analyse)
    assert r["accepte"] is True
    print("  OK : selection valide retenue")


def test_gate_refuse_si_ic80_trop_large():
    """Si l'IC80 depasse le seuil, refus meme si tout le reste est bon."""
    analyse = {
        "n": 20,
        "p_estime": 0.60,
        "p_implicite_sans_marge": 0.52,
        "ev_p20": 0.02,
        "spread_modeles": 0.03,
        "stable_sensibilite": True,
        "ic80_largeur": IC80_LARGEUR_MAX + 0.05,  # trop large
        "contexte": {},
    }
    r = evaluer_selection(analyse)
    assert r["accepte"] is False
    assert any("Critère 7" in x for x in r["raisons_refus"])
    print("  OK : IC80 trop large bloque la selection")


def test_gate_refuse_si_penalite_fait_passer_sous_seuil():
    analyse = {
        "n": 20,
        "p_estime": 0.565,
        "p_implicite_sans_marge": 0.52,
        "ev_p20": 0.01,
        "ic80_largeur": 0.10,
        "contexte": {},
    }
    r = evaluer_selection(analyse)
    assert r["accepte"] is False
    print("  OK : penalite winner's curse protege")


def test_faux_signaux_detectes():
    ctx = {
        "cartons_rouges_recents": 2,
        "cv_mc": 2.5,
        "freq_zero_mc": 0.6,
        "cote_min": 1.20,
        "pagerank_homogene": False,
        "changement_entraineur": True,
        "rotation_probable": True,
    }
    alertes = detecter_faux_signaux(ctx)
    assert len(alertes) == 7, f"attendu 7, obtenu {len(alertes)}"
    print(f"  OK : 7 faux signaux detectes")


def test_faux_signaux_bloque_acceptation():
    analyse = {
        "n": 20,
        "p_estime": 0.60,
        "p_implicite_sans_marge": 0.50,
        "ev_p20": 0.05,
        "ic80_largeur": 0.10,
        "contexte": {"cartons_rouges_recents": 3},
    }
    r = evaluer_selection(analyse)
    assert r["accepte"] is False
    print("  OK : faux signaux bloquent l'acceptation")


if __name__ == "__main__":
    tests = [
        test_ev_over_simple, test_ev_under,
        test_kelly_zero_si_ev_negatif, test_kelly_positif_si_edge,
        test_kelly_prudent_plafonne, test_ev_posterior_bornes,
        test_ic80_present_et_ordonne, test_ic80_serre_pour_n_grand,
        test_edge_simple, test_penalite_winner_curse,
        test_gate_refus_par_defaut, test_gate_accepte_si_tout_bon,
        test_gate_refuse_si_ic80_trop_large,
        test_gate_refuse_si_penalite_fait_passer_sous_seuil,
        test_faux_signaux_detectes, test_faux_signaux_bloque_acceptation,
    ]
    print("=== Tests de la decision (avec IC80) ===\n")
    r = 0
    for t in tests:
        try:
            t(); r += 1
        except Exception as e:
            print(f"  ECHEC : {t.__name__} -> {e}")
    print(f"\n=== {r}/{len(tests)} tests reussis ===")
    if r != len(tests):
        raise SystemExit(1)
