"""Tests du facteur de ligue et style d'equipe."""
from core.league_factor import (
    normaliser_nom, detecter_ligue, facteur_ligue, style_equipe,
    REFERENCE_LIGUES, REFERENCE_GLOBALE,
    SEUIL_OFFENSIF, SEUIL_DEFENSIF,
)


def test_normaliser():
    assert normaliser_nom("Ligue 1") == "ligue 1"
    assert normaliser_nom("PREMIER LEAGUE") == "premier league"
    assert normaliser_nom("Sé rie A") == "se rie a"
    assert normaliser_nom("  La Liga  ") == "la liga"
    assert normaliser_nom(None) == ""
    print("  OK : normalisation")


def test_detecter_ligue_exacte():
    cle, taux = detecter_ligue("Ligue 1")
    assert cle == "ligue 1"
    assert abs(taux - 9.5) < 0.01
    print(f"  OK : Ligue 1 -> {taux}")


def test_detecter_ligue_fuzzy():
    cle, taux = detecter_ligue("Premier League Anglaise")
    assert cle == "premier league"
    print(f"  OK : fuzzy match -> {cle}")


def test_detecter_ligue_inconnue():
    cle, taux = detecter_ligue("Championnat du Botswana")
    assert cle is None
    assert abs(taux - REFERENCE_GLOBALE) < 0.01
    print("  OK : ligue inconnue -> reference globale")


def test_facteur_ligue_faible_exposant():
    """Un exposant 0.3 doit reduire l'ecart entre ligues."""
    f_pl = facteur_ligue("Premier League")["facteur"]
    f_l1 = facteur_ligue("Ligue 1")["facteur"]
    # PL: 10.6/10 = 1.06 -> 1.06^0.3 ~ 1.0176
    # L1: 9.5/10 = 0.95 -> 0.95^0.3 ~ 0.9847
    assert 1.0 < f_pl < 1.05
    assert 0.95 < f_l1 < 1.0
    print(f"  OK : f_PL={f_pl:.4f}, f_L1={f_l1:.4f}")


def test_facteur_ligue_structure():
    r = facteur_ligue("Bundesliga")
    for k in ("competition", "ligue_detectee", "facteur", "connue",
              "ratio_brut", "taux_reference"):
        assert k in r
    print("  OK : structure du facteur")


def test_style_offensif():
    """Une equipe tres offensive doit etre identifiee."""
    # 15 matchs, 4 corners 1h + 6 corners 2h = 10/match, ligue = 10 -> ratio 2
    hist = [
        {"date": f"2026-01-{i+1:02d}", "opponent": f"X{i}", "venue": "H",
         "corners_1h": 4, "corners_2h": 6,
         "resultat_final": "V", "resultat_mt": "N", "cartons_rouges": 0,
         "xg_pour": 1.5, "xg_contre": 0.9,
         "xgot_pour": 1.2, "xgot_contre": 0.7}
        for i in range(15)
    ]
    r = style_equipe(hist, "Ligue 1")  # taux 9.5, attendu 4.75
    # moyenne equipe 10 -> ratio 10/4.75 ~ 2.1 -> offensif
    assert r["label"] == "offensif"
    assert r["ratio"] >= SEUIL_OFFENSIF
    print(f"  OK : style offensif, ratio={r['ratio']:.2f}")


def test_style_defensif():
    hist = [
        {"date": f"2026-01-{i+1:02d}", "opponent": f"X{i}", "venue": "H",
         "corners_1h": 1, "corners_2h": 2,
         "resultat_final": "N", "resultat_mt": "N", "cartons_rouges": 0,
         "xg_pour": 0.8, "xg_contre": 1.2,
         "xgot_pour": 0.6, "xgot_contre": 1.0}
        for i in range(15)
    ]
    r = style_equipe(hist, "Ligue 1")  # attendu 4.75, moyenne 3
    assert r["label"] == "defensif"
    assert r["ratio"] <= SEUIL_DEFENSIF
    print(f"  OK : style defensif, ratio={r['ratio']:.2f}")


def test_style_equilibre():
    hist = [
        {"date": f"2026-01-{i+1:02d}", "opponent": f"X{i}", "venue": "H",
         "corners_1h": 2, "corners_2h": 3,
         "resultat_final": "N", "resultat_mt": "N", "cartons_rouges": 0,
         "xg_pour": 1.2, "xg_contre": 1.2,
         "xgot_pour": 1.0, "xgot_contre": 1.0}
        for i in range(15)
    ]
    r = style_equipe(hist, "Ligue 1")  # attendu 4.75, moyenne 5
    assert r["label"] == "equilibre"
    print(f"  OK : style equilibre, ratio={r['ratio']:.2f}")


def test_style_hist_vide():
    r = style_equipe([], "Ligue 1")
    assert r["label"] == "inconnu"
    print("  OK : hist vide gere")


if __name__ == "__main__":
    tests = [
        test_normaliser,
        test_detecter_ligue_exacte,
        test_detecter_ligue_fuzzy,
        test_detecter_ligue_inconnue,
        test_facteur_ligue_faible_exposant,
        test_facteur_ligue_structure,
        test_style_offensif,
        test_style_defensif,
        test_style_equilibre,
        test_style_hist_vide,
    ]
    print("=== Tests facteur ligue + style ===\n")
    r = 0
    for t in tests:
        try:
            t(); r += 1
        except Exception as e:
            print(f"  ECHEC : {t.__name__} -> {e}")
    print(f"\n=== {r}/{len(tests)} tests reussis ===")
    if r != len(tests):
        raise SystemExit(1)
