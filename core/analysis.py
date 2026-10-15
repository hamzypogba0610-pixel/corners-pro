"""
Orchestrateur : de la saisie au verdict.
Prend le dict du formulaire et produit un rapport complet.
Reference : MULTICORNER 360 v2, sections 4, 8, 9, 10.
"""
import numpy as np
from data.schema import valider_match, ValidationError
from core.hierarchical import build_mc_global, build_pmf_equipe
from core.copula import joint_avec_copule
from core.product_law import (
    product_pmf, convolve, p_over, p_under,
    moments, p_zero, margin_removal_proportional,
    margin_removal_power,
)
from core.pagerank import rapport_pagerank
from core.xg_context import entrainer_sur_les_deux_equipes
from decision.ev import (
    ev_over, ev_under, ev_posterior, kelly_prudent,
)
from decision.gate import evaluer_selection, verdict_texte


CHAMPS_HIST = ("date", "opponent", "corners_1h", "corners_2h",
               "resultat_final", "resultat_mt", "cartons_rouges",
               "xg_pour", "xg_contre", "xgot_pour", "xgot_contre")


def parser_historique(texte, venue, equipe, lignes_attendues=15):
    """Convertit un bloc texte CSV en liste de dicts conforme au schema."""
    erreurs = []
    matchs = []
    for i, ligne in enumerate(texte.strip().splitlines(), start=1):
        ligne = ligne.strip()
        if not ligne or ligne.startswith("#"):
            continue
        parts = [p.strip() for p in ligne.split(",")]
        if len(parts) != 11:
            erreurs.append(
                f"Ligne {i} : {len(parts)} valeurs au lieu de 11 "
                f"(attendu : date,adversaire,c1h,c2h,rf,rmt,rouge,xg,xga,xgot,xgota)"
            )
            continue
        try:
            m = {
                "date": parts[0],
                "opponent": parts[1],
                "corners_1h": int(parts[2]),
                "corners_2h": int(parts[3]),
                "resultat_final": parts[4].upper(),
                "resultat_mt": parts[5].upper(),
                "cartons_rouges": int(parts[6]),
                "xg_pour": float(parts[7]),
                "xg_contre": float(parts[8]),
                "xgot_pour": float(parts[9]),
                "xgot_contre": float(parts[10]),
            }
            m["venue"] = venue
            matchs.append(m)
        except (ValueError, IndexError) as e:
            erreurs.append(f"Ligne {i} : conversion impossible ({e})")
    if erreurs:
        raise ValidationError(
            f"Historique {equipe} :\n" + "\n".join(erreurs)
        )
    if len(matchs) != lignes_attendues:
        raise ValidationError(
            f"Historique {equipe} : {len(matchs)} matchs valides "
            f"au lieu de {lignes_attendues}."
        )
    return matchs


def _build_pmf_par_equipe(hist):
    """Construit la loi jointe A1 x A2 pour une equipe, avec copule."""
    pmfs = build_pmf_equipe(hist)
    a1 = pmfs["A1"]
    a2 = pmfs["A2"]
    c1h = [m["corners_1h"] for m in hist]
    c2h = [m["corners_2h"] for m in hist]
    joint, info_cop = joint_avec_copule(a1, a2, c1h, c2h)
    return joint, info_cop, pmfs


def _p_over_under(pmf, ligne):
    return p_over(pmf, ligne), p_under(pmf, ligne)


def _distribution_posteriori(p_centre, n_effectif=20, n_samples=5000, seed=42):
    """Approximation a posteriori : beta centree sur p_centre."""
    p_centre = max(1e-6, min(1 - 1e-6, p_centre))
    a = p_centre * n_effectif
    b = (1 - p_centre) * n_effectif
    rng = np.random.default_rng(seed)
    return rng.beta(a, b, n_samples)


def analyser_match(data):
    """Point d'entree. Prend un dict complet, retourne un rapport."""
    valider_match(data)

    hist_home = data["home_team_history"]
    hist_away = data["away_team_history"]

    # ----- 1. Construction des lois par equipe -----
    joint_home, cop_home, pmfs_home = _build_pmf_par_equipe(hist_home)
    joint_away, cop_away, pmfs_away = _build_pmf_par_equipe(hist_away)

    mc1_pmf = product_pmf(joint_home)
    mc2_pmf = product_pmf(joint_away)
    global_pmf = convolve(mc1_pmf, mc2_pmf)

    # ----- 2. Statistiques descriptives (toutes les cles necessaires) -----
    stats = {
        "MC1": moments(mc1_pmf),
        "MC2": moments(mc2_pmf),
        "GLOBAL": moments(global_pmf),
        "p_zero_MC1": p_zero(mc1_pmf),
        "p_zero_MC2": p_zero(mc2_pmf),
        "p_zero_GLOBAL": p_zero(global_pmf),
    }

    # ----- 3. PageRank (contexte adversaires) -----
    pr = rapport_pagerank(hist_home, hist_away,
                          data["match"]["home_team"],
                          data["match"]["away_team"])

    # ----- 4. Analyse de chaque marche -----
    marches = {}
    lignes = data["market_lines"]
    pmfs_marche = {"MC1": mc1_pmf, "MC2": mc2_pmf, "GLOBAL": global_pmf}
    zeros_marche = {
        "MC1": stats["p_zero_MC1"],
        "MC2": stats["p_zero_MC2"],
        "GLOBAL": stats["p_zero_GLOBAL"],
    }

    for nom_marche, pmf in pmfs_marche.items():
        ligne = float(lignes[nom_marche]["line"])
        cote_over = float(lignes[nom_marche]["over"])
        cote_under = float(lignes[nom_marche]["under"])

        p_over_val, p_under_val = _p_over_under(pmf, ligne)

        pi_over_prop, pi_under_prop, marge = margin_removal_proportional(
            cote_over, cote_under
        )
        pi_over_pow, pi_under_pow, _ = margin_removal_power(
            cote_over, cote_under
        )

        post_over = _distribution_posteriori(p_over_val)
        ev_over_info = ev_posterior(post_over, cote_over)
        post_under = _distribution_posteriori(p_under_val)
        ev_under_info = ev_posterior(post_under, cote_under)

        # Cartons rouges : on regarde UNIQUEMENT les 5 derniers matchs
        # de CHAQUE equipe. Un rouge sur 30 matchs n'est pas un signal.
        rouges_home_5 = sum(m["cartons_rouges"] for m in hist_home[:5])
        rouges_away_5 = sum(m["cartons_rouges"] for m in hist_away[:5])
        rouges_recents = max(rouges_home_5, rouges_away_5)

        freq_zero = zeros_marche[nom_marche]

        ctx = {
            "cartons_rouges_recents": rouges_recents,
            "freq_zero_mc": freq_zero,
            "pagerank_homogene": pr["homogene"],
            "cote_min": min(cote_over, cote_under),
        }

        analyse_over = {
            "n": 15,
            "p_estime": p_over_val,
            "p_implicite_sans_marge": pi_over_prop,
            "ev_p20": ev_over_info["ev_p20"],
            "spread_modeles": abs(pi_over_prop - pi_over_pow),
            "stable_sensibilite": True,
            "contexte": ctx,
        }
        analyse_under = {
            "n": 15,
            "p_estime": p_under_val,
            "p_implicite_sans_marge": pi_under_prop,
            "ev_p20": ev_under_info["ev_p20"],
            "spread_modeles": abs(pi_under_prop - pi_under_pow),
            "stable_sensibilite": True,
            "contexte": ctx,
        }
        res_over = evaluer_selection(analyse_over)
        res_under = evaluer_selection(analyse_under)

        cotes = {
            "over": {
                "p": p_over_val,
                "cote": cote_over,
                "p_impl_prop": pi_over_prop,
                "p_impl_pow": pi_over_pow,
                "ev_info": ev_over_info,
                "gate": res_over,
                "kelly": kelly_prudent(
                    max(0.0, ev_over_info["p_gain_p20"]), cote_over
                ),
            },
            "under": {
                "p": p_under_val,
                "cote": cote_under,
                "p_impl_prop": pi_under_prop,
                "p_impl_pow": pi_under_pow,
                "ev_info": ev_under_info,
                "gate": res_under,
                "kelly": kelly_prudent(
                    max(0.0, ev_under_info["p_gain_p20"]), cote_under
                ),
            },
        }

        marches[nom_marche] = {
            "ligne": ligne,
            "cotes": cotes,
            "marge_totale": marge,
            "ecart_methode_marge": abs(pi_over_prop - pi_over_pow),
        }

    # ----- 5. Synthese -----
    selections_acceptees = []
    for nom_marche, info in marches.items():
        for cote in ("over", "under"):
            if info["cotes"][cote]["gate"]["accepte"]:
                selections_acceptees.append(
                    f"{nom_marche} {cote.upper()} {info['ligne']}"
                )

    verdict = (
        "SELECTION : " + " | ".join(selections_acceptees)
        if selections_acceptees
        else "AUCUNE SELECTION SUFFISAMMENT SOLIDE POUR CE MARCHE."
    )

    return {
        "match": data["match"],
        "stats": stats,
        "marches": marches,
        "pagerank": pr,
        "copule_home": cop_home,
        "copule_away": cop_away,
        "selections_acceptees": selections_acceptees,
        "verdict": verdict,
}
