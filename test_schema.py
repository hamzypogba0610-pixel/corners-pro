"""
Tests du gardien de donnees (data/schema.py) - version 4.
Repartition 10 matchs au lieu + 5 globaux ('N').
"""
from data.schema import valider_match, ValidationError


def _hist(venue_principal="H"):
    """10 matchs au lieu principal + 5 globaux."""
    h = []
    for i in range(10):
        h.append({
            "date": f"2026-01-{i+1:02d}",
            "opponent": f"T{i}",
            "venue": venue_principal,
            "corners_1h": 3, "corners_2h": 4,
            "resultat_final": "V", "resultat_mt": "N", "cartons_rouges": 0,
            "xg_pour": 1.5, "xg_contre": 0.9,
            "xgot_pour": 1.2, "xgot_contre": 0.7,
        })
    for i in range(5):
        h.append({
            "date": f"2026-02-{i+1:02d}",
            "opponent": f"G{i}",
            "venue": "N",
            "corners_1h": 2, "corners_2h": 3,
            "resultat_final": "N", "resultat_mt": "N", "cartons_rouges": 0,
            "xg_pour": 1.3, "xg_contre": 1.1,
            "xgot_pour": 1.0, "xgot_contre": 0.9,
        })
    return h


def match_bidon_ok():
    return {
        "match": {
            "date": "2026-03-01",
            "competition": "Ligue 1",
            "home_team": "Paris",
            "away_team": "Lyon",
            "mode": "COMPLET",
        },
        "home_team_history": _hist("H"),
        "away_team_history": _hist("A"),
        "market_lines": {
            "MC1": {"line": 12.5, "over": 1.90, "under": 1.90},
            "MC2": {"line": 12.5, "over": 1.85, "under": 1.95},
            "GLOBAL": {"line": 25.5, "over": 1.88, "under": 1.92},
        },
        "odds_1x2": {"home": 2.10, "draw": 3.40, "away": 3.20},
    }


def test_valide_ok():
    assert valider_match(match_bidon_ok()) is True
    print("  OK : match valide (10 lieu + 5 globaux)")


def test_date_manquante():
    d = match_bidon_ok(); del d["match"]["date"]
    try:
        valider_match(d); raise AssertionError("Devrait refuser")
    except ValidationError:
        print("  OK : date manquante refusee")


def test_date_mal_formee():
    d = match_bidon_ok(); d["match"]["date"] = "01/02/2026"
    try:
        valider_match(d); raise AssertionError("Devrait refuser")
    except ValidationError:
        print("  OK : date mal formee refusee")


def test_equipes_identiques():
    d = match_bidon_ok(); d["match"]["away_team"] = "Paris"
    try:
        valider_match(d); raise AssertionError("Devrait refuser")
    except ValidationError:
        print("  OK : equipes identiques refusees")


def test_moins_de_15_matchs():
    d = match_bidon_ok(); d["home_team_history"] = d["home_team_history"][:10]
    try:
        valider_match(d); raise AssertionError("Devrait refuser")
    except ValidationError:
        print("  OK : historique incomplet refuse")


def test_moins_de_10_au_lieu():
    """Seulement 5 matchs 'H' + 10 globaux -> refuse."""
    d = match_bidon_ok()
    hist = d["home_team_history"]
    for i in range(5, 10):
        hist[i]["venue"] = "N"  # 5 H + 10 N
    try:
        valider_match(d); raise AssertionError("Devrait refuser")
    except ValidationError as e:
        assert "10" in str(e)
        print(f"  OK : moins de 10 au lieu refuse")


def test_venue_inverse_refusee():
    """Un match 'A' dans l'historique domicile -> refuse."""
    d = match_bidon_ok()
    d["home_team_history"][0]["venue"] = "A"
    try:
        valider_match(d); raise AssertionError("Devrait refuser")
    except ValidationError:
        print("  OK : venue inverse refuse")


def test_venue_N_acceptee():
    """Un match 'N' est accepte comme match global."""
    d = match_bidon_ok()
    assert valider_match(d) is True
    print("  OK : venue 'N' (global) acceptee")


def test_corner_negatif():
    d = match_bidon_ok(); d["home_team_history"][0]["corners_1h"] = -1
    try:
        valider_match(d); raise AssertionError("Devrait refuser")
    except ValidationError:
        print("  OK : corner negatif refuse")


def test_cote_invalide():
    d = match_bidon_ok(); d["market_lines"]["MC1"]["over"] = 0.95
    try:
        valider_match(d); raise AssertionError("Devrait refuser")
    except ValidationError:
        print("  OK : cote <= 1 refusee")


def test_resultat_invalide():
    d = match_bidon_ok(); d["home_team_history"][0]["resultat_final"] = "X"
    try:
        valider_match(d); raise AssertionError("Devrait refuser")
    except ValidationError:
        print("  OK : resultat invalide refuse")


def test_xg_negatif():
    d = match_bidon_ok(); d["home_team_history"][4]["xg_contre"] = -0.3
    try:
        valider_match(d); raise AssertionError("Devrait refuser")
    except ValidationError:
        print("  OK : xg negatif refuse")


def test_odds_1x2_manquant():
    d = match_bidon_ok(); del d["odds_1x2"]
    try:
        valider_match(d); raise AssertionError("Devrait refuser")
    except ValidationError:
        print("  OK : odds_1x2 manquantes refusees")


if __name__ == "__main__":
    tests = [
        test_valide_ok,
        test_date_manquante,
        test_date_mal_formee,
        test_equipes_identiques,
        test_moins_de_15_matchs,
        test_moins_de_10_au_lieu,
        test_venue_inverse_refusee,
        test_venue_N_acceptee,
        test_corner_negatif,
        test_cote_invalide,
        test_resultat_invalide,
        test_xg_negatif,
        test_odds_1x2_manquant,
    ]
    print("=== Tests schema v4 (10 + 5) ===\n")
    r = 0
    for t in tests:
        try:
            t(); r += 1
        except Exception as e:
            print(f"  ECHEC : {t.__name__} -> {e}")
    print(f"\n=== {r}/{len(tests)} tests reussis ===")
    if r != len(tests):
        raise SystemExit(1)
