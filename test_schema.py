"""
Tests du gardien de donnees (data/schema.py).
On essaie de lui donner des donnees pourries et on verifie qu'il refuse.
"""
from data.schema import valider_match, ValidationError


def match_bidon_ok():
    """Un match complet et valide, qu'on va ensuite saboter pour les tests."""
    home_hist = [
        {"date": "2026-01-01", "opponent": "TeamA", "venue": "H",
         "corners_1h": 3, "corners_2h": 4}
    ] * 15
    away_hist = [
        {"date": "2026-01-01", "opponent": "TeamB", "venue": "A",
         "corners_1h": 2, "corners_2h": 3}
    ] * 15
    return {
        "match": {
            "date": "2026-02-01",
            "competition": "Ligue 1",
            "home_team": "Paris",
            "away_team": "Lyon",
            "mode": "COMPLET",
        },
        "home_team_history": home_hist,
        "away_team_history": away_hist,
        "market_lines": {
            "MC1": {"line": 12.5, "over": 1.90, "under": 1.90},
            "MC2": {"line": 12.5, "over": 1.85, "under": 1.95},
            "GLOBAL": {"line": 25.5, "over": 1.88, "under": 1.92},
        },
    }


def test_valide_ok():
    d = match_bidon_ok()
    assert valider_match(d) is True
    print("  OK : un match valide passe")


def test_date_manquante():
    d = match_bidon_ok()
    del d["match"]["date"]
    try:
        valider_match(d)
        raise AssertionError("Devrait refuser une date manquante")
    except ValidationError as e:
        print(f"  OK : date manquante refusee ({e})")


def test_date_mal_formee():
    d = match_bidon_ok()
    d["match"]["date"] = "01/02/2026"
    try:
        valider_match(d)
        raise AssertionError("Devrait refuser une date mal formee")
    except ValidationError:
        print("  OK : date mal formee refusee")


def test_equipes_identiques():
    d = match_bidon_ok()
    d["match"]["away_team"] = "Paris"
    try:
        valider_match(d)
        raise AssertionError("Devrait refuser les equipes identiques")
    except ValidationError:
        print("  OK : equipes identiques refusees")


def test_moins_de_15_matchs():
    d = match_bidon_ok()
    d["home_team_history"] = d["home_team_history"][:10]
    try:
        valider_match(d)
        raise AssertionError("Devrait refuser moins de 15 matchs")
    except ValidationError as e:
        print(f"  OK : historique incomplet refuse ({e})")


def test_corner_negatif():
    d = match_bidon_ok()
    d["home_team_history"][0]["corners_1h"] = -1
    try:
        valider_match(d)
        raise AssertionError("Devrait refuser corner negatif")
    except ValidationError:
        print("  OK : corner negatif refuse")


def test_venue_inverse():
    """Un match 'away' dans l'historique 'home' doit etre refuse."""
    d = match_bidon_ok()
    d["home_team_history"][0]["venue"] = "A"
    try:
        valider_match(d)
        raise AssertionError("Devrait refuser mauvais venue")
    except ValidationError:
        print("  OK : venue incorrect refuse")


def test_cote_invalide():
    d = match_bidon_ok()
    d["market_lines"]["MC1"]["over"] = 0.95
    try:
        valider_match(d)
        raise AssertionError("Devrait refuser cote <= 1")
    except ValidationError:
        print("  OK : cote <= 1 refusee")


def test_marche_manquant():
    d = match_bidon_ok()
    del d["market_lines"]["GLOBAL"]
    try:
        valider_match(d)
        raise AssertionError("Devrait refuser marche manquant")
    except ValidationError:
        print("  OK : marche GLOBAL manquant refuse")


def test_adversaire_vide():
    d = match_bidon_ok()
    d["home_team_history"][3]["opponent"] = ""
    try:
        valider_match(d)
        raise AssertionError("Devrait refuser adversaire vide")
    except ValidationError:
        print("  OK : adversaire vide refuse")


if __name__ == "__main__":
    tests = [
        test_valide_ok,
        test_date_manquante,
        test_date_mal_formee,
        test_equipes_identiques,
        test_moins_de_15_matchs,
        test_corner_negatif,
        test_venue_inverse,
        test_cote_invalide,
        test_marche_manquant,
        test_adversaire_vide,
    ]
    print("=== Tests du gardien de donnees ===\n")
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
