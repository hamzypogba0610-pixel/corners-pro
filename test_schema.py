"""
Tests du gardien de donnees (data/schema.py) - version 2.
"""
from data.schema import valider_match, ValidationError


def _hist(venue="H"):
    return [
        {"date": f"2026-01-{i+1:02d}", "opponent": f"Team{i}",
         "venue": venue, "corners_1h": 3, "corners_2h": 4,
         "resultat_final": "V", "resultat_mt": "N", "cartons_rouges": 0}
        for i in range(15)
    ]


def match_bidon_ok():
    return {
        "match": {
            "date": "2026-02-01",
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
    print("  OK : un match valide passe")


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


def test_corner_negatif():
    d = match_bidon_ok(); d["home_team_history"][0]["corners_1h"] = -1
    try:
        valider_match(d); raise AssertionError("Devrait refuser")
    except ValidationError:
        print("  OK : corner negatif refuse")


def test_venue_inverse():
    d = match_bidon_ok(); d["home_team_history"][0]["venue"] = "A"
    try:
        valider_match(d); raise AssertionError("Devrait refuser")
    except ValidationError:
        print("  OK : venue incorrect refuse")


def test_cote_invalide():
    d = match_bidon_ok(); d["market_lines"]["MC1"]["over"] = 0.95
    try:
        valider_match(d); raise AssertionError("Devrait refuser")
    except ValidationError:
        print("  OK : cote <= 1 refusee")


def test_marche_manquant():
    d = match_bidon_ok(); del d["market_lines"]["GLOBAL"]
    try:
        valider_match(d); raise AssertionError("Devrait refuser")
    except ValidationError:
        print("  OK : marche GLOBAL manquant refuse")


def test_adversaire_vide():
    d = match_bidon_ok(); d["home_team_history"][3]["opponent"] = ""
    try:
        valider_match(d); raise AssertionError("Devrait refuser")
    except ValidationError:
        print("  OK : adversaire vide refuse")


def test_resultat_invalide():
    d = match_bidon_ok(); d["home_team_history"][0]["resultat_final"] = "X"
    try:
        valider_match(d); raise AssertionError("Devrait refuser")
    except ValidationError:
        print("  OK : resultat invalide refuse")


def test_resultat_mt_manquant():
    d = match_bidon_ok(); del d["home_team_history"][0]["resultat_mt"]
    try:
        valider_match(d); raise AssertionError("Devrait refuser")
    except ValidationError:
        print("  OK : resultat_mt manquant refuse")


def test_cartons_rouges_negatif():
    d = match_bidon_ok(); d["home_team_history"][2]["cartons_rouges"] = -1
    try:
        valider_match(d); raise AssertionError("Devrait refuser")
    except ValidationError:
        print("  OK : cartons rouges negatifs refuses")


def test_odds_1x2_manquant():
    d = match_bidon_ok(); del d["odds_1x2"]
    try:
        valider_match(d); raise AssertionError("Devrait refuser")
    except ValidationError:
        print("  OK : odds_1x2 manquantes refusees")


def test_odds_1x2_invalide():
    d = match_bidon_ok(); d["odds_1x2"]["home"] = 0.95
    try:
        valider_match(d); raise AssertionError("Devrait refuser")
    except ValidationError:
        print("  OK : cote 1X2 <= 1 refusee")


if __name__ == "__main__":
    tests = [
        test_valide_ok, test_date_manquante, test_date_mal_formee,
        test_equipes_identiques, test_moins_de_15_matchs,
        test_corner_negatif, test_venue_inverse, test_cote_invalide,
        test_marche_manquant, test_adversaire_vide,
        test_resultat_invalide, test_resultat_mt_manquant,
        test_cartons_rouges_negatif, test_odds_1x2_manquant,
        test_odds_1x2_invalide,
    ]
    print("=== Tests du gardien v2 ===\n")
    r = 0
    for t in tests:
        try:
            t(); r += 1
        except Exception as e:
            print(f"  ECHEC : {t.__name__} -> {e}")
    print(f"\n=== {r}/{len(tests)} tests reussis ===")
    if r != len(tests):
        raise SystemExit(1)
