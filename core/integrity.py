"""
Couche d'integrite : verifie que chaque couche du modele a fait son travail.
Reference : MULTICORNER 360 v2, sections 3, 4, 6.
"""
import numpy as np


TOLERANCE_PMF = 1e-6
TOLERANCE_COHERENCE = 1e-6


def _est_fini(x):
    if x is None:
        return False
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def verifier_pmf(pmf_dict, nom="pmf", tol=TOLERANCE_PMF):
    """Verifie qu'une pmf (dict) somme a 1, sans valeur negative ni NaN."""
    problemes = []
    if pmf_dict is None or len(pmf_dict) == 0:
        problemes.append(f"{nom} : vide")
        return problemes
    total = 0.0
    for k, v in pmf_dict.items():
        if not _est_fini(v):
            problemes.append(f"{nom}[{k}] : valeur non finie ({v!r})")
            continue
        if v < 0:
            problemes.append(f"{nom}[{k}] : proba negative ({v})")
        total += float(v)
    if abs(total - 1.0) > tol:
        problemes.append(f"{nom} : somme = {total:.8f} (attendu 1)")
    return problemes


def verifier_pmf_array(arr, nom="pmf", tol=TOLERANCE_PMF):
    """Verifie qu'un array numpy est une pmf valide."""
    problemes = []
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        problemes.append(f"{nom} : vide")
        return problemes
    if not np.all(np.isfinite(a)):
        problemes.append(f"{nom} : contient NaN ou infini")
    if np.any(a < -tol):
        problemes.append(f"{nom} : contient des valeurs negatives")
    s = float(a.sum())
    if abs(s - 1.0) > tol:
        problemes.append(f"{nom} : somme = {s:.8f} (attendu 1)")
    return problemes


def verifier_proba(p, nom="p"):
    problemes = []
    if not _est_fini(p):
        problemes.append(f"{nom} : non fini ({p!r})")
        return problemes
    p = float(p)
    if p < 0 or p > 1:
        problemes.append(f"{nom} : hors [0,1] ({p})")
    return problemes


def verifier_borne(x, borne_min, borne_max, nom):
    problemes = []
    if not _est_fini(x):
        problemes.append(f"{nom} : non fini")
        return problemes
    x = float(x)
    if x < borne_min or x > borne_max:
        problemes.append(
            f"{nom} : hors [{borne_min}, {borne_max}] ({x})"
        )
    return problemes


def verifier_coherence_convolution(mc1_pmf, mc2_pmf, global_pmf,
                                    tol=TOLERANCE_COHERENCE):
    """E[GLOBAL] doit egaler E[MC1] + E[MC2] a epsilon pres."""
    problemes = []
    def moy(pmf):
        return sum(k * v for k, v in pmf.items())
    try:
        m1 = moy(mc1_pmf)
        m2 = moy(mc2_pmf)
        mg = moy(global_pmf)
    except Exception as e:
        problemes.append(f"coherence convolution : {e}")
        return problemes
    ecart = abs(mg - (m1 + m2))
    if ecart > tol:
        problemes.append(
            f"coherence convolution : E[GLOBAL]={mg:.6f} != "
            f"E[MC1]+E[MC2]={m1 + m2:.6f} (ecart {ecart:.6g})"
        )
    return problemes


def verifier_parametres_nb(alpha, beta, nom="NB"):
    problemes = []
    problemes.extend(verifier_borne(alpha, 1e-9, 1e6, f"{nom}.alpha"))
    problemes.extend(verifier_borne(beta, 1e-9, 1e6, f"{nom}.beta"))
    return problemes


def verifier_parametres_zinb(alpha, beta, pi, nom="ZINB"):
    problemes = verifier_parametres_nb(alpha, beta, nom)
    problemes.extend(verifier_borne(pi, 0.0, 0.5, f"{nom}.pi"))
    return problemes


def verifier_parametres_copule(theta, tau, nom="copule"):
    problemes = []
    problemes.extend(verifier_borne(theta, -30.0, 30.0, f"{nom}.theta"))
    problemes.extend(verifier_borne(tau, -1.0, 1.0, f"{nom}.tau"))
    return problemes


def verifier_facteurs(facteur_ligue, facteur_home_2h, facteur_away_2h,
                     nom="facteurs"):
    problemes = []
    problemes.extend(
        verifier_borne(facteur_ligue, 0.5, 2.0, f"{nom}.ligue")
    )
    problemes.extend(
        verifier_borne(facteur_home_2h, 0.5, 2.0, f"{nom}.home_2h")
    )
    problemes.extend(
        verifier_borne(facteur_away_2h, 0.5, 2.0, f"{nom}.away_2h")
    )
    return problemes


def verifier_rapport_complet(rapport):
    """
    Verification de bout en bout du rapport d'analyse.
    Retourne dict {ok, problemes_bloquants, avertissements}.
    """
    bloquants = []
    avertissements = []

    if not isinstance(rapport, dict):
        return {
            "ok": False,
            "problemes_bloquants": ["rapport non conforme"],
            "avertissements": [],
        }

    # Stats
    for cle in ("MC1", "MC2", "GLOBAL"):
        if cle not in rapport.get("stats", {}):
            bloquants.append(f"stats.{cle} manquant")

    if "marches" not in rapport:
        bloquants.append("section marches manquante")
        return {
            "ok": False,
            "problemes_bloquants": bloquants,
            "avertissements": avertissements,
        }

    # Chaque marche, chaque cote
    for nom_marche, info in rapport["marches"].items():
        for cote_nom in ("over", "under"):
            c = info.get("cotes", {}).get(cote_nom)
            if not c:
                bloquants.append(f"{nom_marche}.{cote_nom} manquant")
                continue
            for cle in ("p", "cote", "p_impl_prop", "kelly"):
                if cle not in c:
                    bloquants.append(
                        f"{nom_marche}.{cote_nom}.{cle} manquant"
                    )
            if "p" in c:
                bloquants.extend(verifier_proba(
                    c["p"], f"{nom_marche}.{cote_nom}.p"
                ))
            if "cote" in c:
                bloquants.extend(verifier_borne(
                    c["cote"], 1.0, 1000.0,
                    f"{nom_marche}.{cote_nom}.cote"
                ))
            if "kelly" in c:
                bloquants.extend(verifier_borne(
                    c["kelly"], 0.0, 1.0,
                    f"{nom_marche}.{cote_nom}.kelly"
                ))

    # Copules (avertissements seulement)
    for nom in ("copule_home", "copule_away"):
        cop = rapport.get(nom)
        if cop and "theta" in cop and "tau" in cop:
            avertissements.extend(verifier_parametres_copule(
                cop["theta"], cop["tau"], nom
            ))

    # Facteurs (avertissements seulement)
    se = rapport.get("score_effect") or {}
    ligue = rapport.get("ligue") or {}
    avertissements.extend(verifier_facteurs(
        ligue.get("facteur", 1.0),
        se.get("facteur_home_2h", 1.0),
        se.get("facteur_away_2h", 1.0),
        "facteurs",
    ))

    return {
        "ok": len(bloquants) == 0,
        "problemes_bloquants": bloquants,
        "avertissements": avertissements,
}
