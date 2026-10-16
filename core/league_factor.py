"""
Facteur de ligue et style d'equipe.
Reference : MULTICORNER 360 v2, section 4 (couche contextuelle).

Table HYPOTHESE : taux approximatifs de corners par match selon les
championnats. Sources publiques (moyennes 2023-2024, Understat/FBref).
Ces valeurs sont INDICATIVES et servent de repere pour normaliser entre
championnats. Elles ne remplacent PAS les donnees saisies par l'utilisateur.
"""
import unicodedata


REFERENCE_LIGUES = {
    "premier league": 10.6,
    "championship": 10.9,
    "la liga": 9.7,
    "segunda division": 9.3,
    "serie a": 10.1,
    "serie b": 9.6,
    "bundesliga": 10.3,
    "2. bundesliga": 10.0,
    "ligue 1": 9.5,
    "ligue 2": 9.2,
    "eredivisie": 10.4,
    "primeira liga": 10.0,
    "jupiler pro league": 10.0,
    "super lig": 10.6,
    "liga mx": 10.2,
    "mls": 10.0,
    "brasileirao": 10.5,
    "argentine": 9.8,
    "champions league": 10.2,
    "europa league": 10.0,
    "conference league": 10.0,
    "coupe du monde": 10.0,
    "euro": 10.0,
    "can": 9.5,
}

REFERENCE_GLOBALE = 10.0
EXPOSANT_DEFAUT = 0.3

SEUIL_OFFENSIF = 1.15
SEUIL_DEFENSIF = 0.85


def normaliser_nom(s):
    """Normalise un nom (minuscules, sans accents, espaces multiples)."""
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )
    s = " ".join(s.split())
    return s


def detecter_ligue(competition):
    """
    Fuzzy match sur le nom de competition.
    Retourne (cle_trouvee, taux) ou (None, REFERENCE_GLOBALE).
    """
    nom = normaliser_nom(competition)
    if not nom:
        return None, REFERENCE_GLOBALE
    if nom in REFERENCE_LIGUES:
        return nom, REFERENCE_LIGUES[nom]
    # Recherche par sous-chaine, en privilegiant la cle la plus longue
    candidats = sorted(REFERENCE_LIGUES.keys(), key=len, reverse=True)
    for cle in candidats:
        if cle in nom or nom in cle:
            return cle, REFERENCE_LIGUES[cle]
    return None, REFERENCE_GLOBALE


def facteur_ligue(competition, exposant=EXPOSANT_DEFAUT):
    """
    Facteur multiplicatif applique a l'intensite de corners.
    Utilise un exposant < 1 pour ne pas ecraser les donnees saisies.
    """
    cle, taux = detecter_ligue(competition)
    ratio_brut = taux / REFERENCE_GLOBALE
    facteur = ratio_brut ** float(exposant)
    return {
        "competition": competition,
        "ligue_detectee": cle,
        "taux_reference": float(taux),
        "reference_globale": float(REFERENCE_GLOBALE),
        "ratio_brut": float(ratio_brut),
        "exposant": float(exposant),
        "facteur": float(facteur),
        "connue": cle is not None,
    }


def style_equipe(hist, competition):
    """
    Evalue le style offensif/defensif d'une equipe.
    Compare sa moyenne de corners par match au taux de reference de sa ligue.
    Retourne dict avec moyenne_equipe, moyenne_attendue, ratio, label.
    """
    if not hist:
        return {
            "moyenne_equipe": 0.0,
            "moyenne_attendue": REFERENCE_GLOBALE / 2.0,
            "ratio": 1.0,
            "label": "inconnu",
        }
    # Moyenne de corners par match pour l'equipe (1h + 2h)
    total_par_match = [m["corners_1h"] + m["corners_2h"] for m in hist]
    moy_equipe = sum(total_par_match) / len(total_par_match)
    # Attendue : ligue (total des 2 equipes) / 2
    _, taux_ligue = detecter_ligue(competition)
    moy_attendue = taux_ligue / 2.0
    ratio = moy_equipe / moy_attendue if moy_attendue > 0 else 1.0
    if ratio >= SEUIL_OFFENSIF:
        label = "offensif"
    elif ratio <= SEUIL_DEFENSIF:
        label = "defensif"
    else:
        label = "equilibre"
    return {
        "moyenne_equipe": float(moy_equipe),
        "moyenne_attendue": float(moy_attendue),
        "ratio": float(ratio),
        "label": label,
        "n": len(hist),
}
