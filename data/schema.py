"""
Schema de validation des donnees saisies par l'utilisateur.
Version 4 : repartition 10 matchs au lieu specifie + 5 globaux.
"""
import re
from datetime import datetime


DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
MARKETS_OK = ("MC1", "MC2", "GLOBAL")
N_MATCHS_REQUIS = 15
N_MATCHS_LIEU = 10        # les 10 premiers doivent etre au lieu specifie
N_MATCHS_GLOBAUX = 5      # les 5 suivants sont globaux ('N')
RESULTATS_OK = ("V", "N", "D")
VENUES_VALIDES = ("H", "A", "N")


class ValidationError(Exception):
    pass


def _check_date(s, champ):
    if not isinstance(s, str) or not DATE_RE.match(s):
        raise ValidationError(
            f"{champ} : date invalide '{s}' (format YYYY-MM-DD attendu)"
        )
    try:
        datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        raise ValidationError(f"{champ} : date inexistante '{s}'")


def _check_entier_positif(v, champ):
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValidationError(f"{champ} : doit etre un entier, recu {v!r}")
    if v < 0:
        raise ValidationError(f"{champ} : ne peut pas etre negatif ({v})")


def _check_float_positif(v, champ):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValidationError(f"{champ} : doit etre un nombre, recu {v!r}")
    if v < 0:
        raise ValidationError(f"{champ} : ne peut pas etre negatif ({v})")


def _check_non_vide(s, champ):
    if not isinstance(s, str) or not s.strip():
        raise ValidationError(f"{champ} : ne peut pas etre vide")


def _check_resultat(v, champ):
    if v not in RESULTATS_OK:
        raise ValidationError(
            f"{champ} : doit etre 'V', 'N' ou 'D', recu {v!r}"
        )


def _check_history(hist, champ, venue_attendu, n=N_MATCHS_REQUIS):
    if not isinstance(hist, list):
        raise ValidationError(f"{champ} : doit etre une liste")
    if len(hist) != n:
        raise ValidationError(f"{champ} : {len(hist)} matchs au lieu de {n}")

    n_lieu = 0
    for i, m in enumerate(hist):
        prefix = f"{champ}[{i}]"
        if not isinstance(m, dict):
            raise ValidationError(f"{prefix} : doit etre un objet")
        _check_date(m.get("date"), f"{prefix}.date")
        _check_non_vide(m.get("opponent"), f"{prefix}.opponent")

        venue = m.get("venue")
        if venue not in VENUES_VALIDES:
            raise ValidationError(
                f"{prefix}.venue doit etre 'H', 'A' ou 'N', recu {venue!r}"
            )
        if venue == venue_attendu:
            n_lieu += 1
        elif venue != "N":
            raise ValidationError(
                f"{prefix}.venue doit valoir '{venue_attendu}' ou 'N', "
                f"recu {venue!r}"
            )

        _check_entier_positif(m.get("corners_1h"), f"{prefix}.corners_1h")
        _check_entier_positif(m.get("corners_2h"), f"{prefix}.corners_2h")
        _check_resultat(m.get("resultat_final"), f"{prefix}.resultat_final")
        _check_resultat(m.get("resultat_mt"), f"{prefix}.resultat_mt")
        _check_entier_positif(m.get("cartons_rouges"), f"{prefix}.cartons_rouges")
        _check_float_positif(m.get("xg_pour"), f"{prefix}.xg_pour")
        _check_float_positif(m.get("xg_contre"), f"{prefix}.xg_contre")
        _check_float_positif(m.get("xgot_pour"), f"{prefix}.xgot_pour")
        _check_float_positif(m.get("xgot_contre"), f"{prefix}.xgot_contre")

    if n_lieu < N_MATCHS_LIEU:
        raise ValidationError(
            f"{champ} : au moins {N_MATCHS_LIEU} matchs doivent etre au lieu "
            f"'{venue_attendu}', trouve {n_lieu}"
        )


def _check_lines(lines, champ):
    if not isinstance(lines, dict):
        raise ValidationError(f"{champ} : doit etre un objet")
    for market in MARKETS_OK:
        if market not in lines:
            raise ValidationError(f"{champ} : marche '{market}' manquant")
        v = lines[market]
        if not isinstance(v, dict):
            raise ValidationError(f"{champ}.{market} : doit etre un objet")
        for k in ("line", "over", "under"):
            if k not in v:
                raise ValidationError(f"{champ}.{market} : champ '{k}' manquant")
        if not isinstance(v["line"], (int, float)) or v["line"] < 0:
            raise ValidationError(
                f"{champ}.{market}.line invalide : {v['line']!r}"
            )
        if not isinstance(v["over"], (int, float)) or v["over"] <= 1.0:
            raise ValidationError(
                f"{champ}.{market}.over doit etre > 1 (recu {v['over']!r})"
            )
        if not isinstance(v["under"], (int, float)) or v["under"] <= 1.0:
            raise ValidationError(
                f"{champ}.{market}.under doit etre > 1 (recu {v['under']!r})"
            )


def _check_odds_1x2(odds, champ):
    if not isinstance(odds, dict):
        raise ValidationError(f"{champ} : doit etre un objet")
    for k in ("home", "draw", "away"):
        if k not in odds:
            raise ValidationError(f"{champ} : champ '{k}' manquant")
        if not isinstance(odds[k], (int, float)) or odds[k] <= 1.0:
            raise ValidationError(
                f"{champ}.{k} doit etre > 1 (recu {odds[k]!r})"
            )


def valider_match(data):
    if not isinstance(data, dict):
        raise ValidationError("Les donnees doivent etre un objet JSON")
    if "match" not in data:
        raise ValidationError("Section 'match' manquante")

    m = data["match"]
    _check_date(m.get("date"), "match.date")
    _check_non_vide(m.get("competition"), "match.competition")
    _check_non_vide(m.get("home_team"), "match.home_team")
    _check_non_vide(m.get("away_team"), "match.away_team")
    if m.get("home_team") == m.get("away_team"):
        raise ValidationError("home_team et away_team ne peuvent pas etre identiques")
    if m.get("mode") not in ("RAPIDE", "COMPLET"):
        raise ValidationError("match.mode doit etre 'RAPIDE' ou 'COMPLET'")

    _check_history(data.get("home_team_history"), "home_team_history", "H")
    _check_history(data.get("away_team_history"), "away_team_history", "A")

    _check_lines(data.get("market_lines"), "market_lines")
    _check_odds_1x2(data.get("odds_1x2"), "odds_1x2")
    return True
