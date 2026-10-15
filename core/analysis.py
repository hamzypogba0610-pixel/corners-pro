"""
Orchestrateur : de la saisie au verdict.
Integre : ponderation temporelle, IC80, effet score, dependance inter-equipes,
inflation de zeros, et test de sensibilite (tornade).
"""
import numpy as np
from data.schema import valider_match, ValidationError
from core.hierarchical import build_pmf_equipe
from core.copula import (
    joint_avec_copule, estimer_correlation_inter_equipes,
    joint_mc_via_copule, global_depuis_joint_mc,
)
from core.product_law import (
    product_pmf, convolve, p_over, p_under,
    moments, p_zero, margin_removal_proportional,
    margin_removal_power,
)
from core.pagerank import rapport_pagerank
from core.score_effect import calculer_facteurs_match
from core.sensitivity import SCENARIOS, perturber_hist
from decision.ev import ev_posterior, kelly_prudent
from decision.gate import evaluer_selection


MARCHES = ("MC1", "MC2", "GLOBAL")
COTES = ("over", "under")


def parser_historique(texte, venue, equipe, lignes_attendues=15):
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


def _distribution_posteriori(p_centre, n_effectif=20, n_samples=5000, seed=42):
    p_centre = max(1e-6, min(1 - 1e-6, p_centre))
    a = p_centre * n_effectif
    b = (1 - p_centre) * n_effectif
    rng = np.random.default_rng(seed)
    return rng.beta(a, b, n_samples)


def _build_pmf_par_equipe(hist, facteur_2h=1.0, demi_vie=8):
    pmfs = build_pmf_equipe(hist, demi_vie=demi_vie, facteur_2h=facteur_2h)
    a1 = pmfs["A1"]; a2 = pmfs["A2"]
    c1h = [m["corners_1h"] for m in hist]
    c2h = [m["corners_2h"] for m in hist]
    joint, info_cop = joint_avec_copule(a1, a2, c1h, c2h)
    return joint, info_cop, pmfs


def _pipeline(data, params=None):
    """
    Pipeline complet avec parametres ajustables.
    params : dict pouvant contenir :
      - demi_vie : int (defaut 8)
      - facteur_moyenne : float (defaut 1.0)
      - retirer_extremes : int (defaut 0)
      - rho_force : float ou None (defaut None = estimation)
    """
    params = params or {}
    demi_vie = params.get("demi_vie", 8)
    facteur_moyenne = params.get("facteur_moyenne", 1.0)
    retirer_extremes = params.get("retirer_extremes", 0)
    rho_force = params.get("rho_force", None)

    hist_home_0 = data["home_team_history"]
    hist_away_0 = data["away_team_history"]

    if facteur_moyenne != 1.0 or retirer_extremes > 0:
        hist_home = perturber_hist(hist_home_0, facteur_moyenne, retirer_extremes)
        hist_away = perturber_hist(hist_away_0, facteur_moyenne, retirer_extremes)
    else:
        hist_home = hist_home_0
        hist_away = hist_away_0

    # Effet score (toujours sur l'historique original)
    score_info = calculer_facteurs_match(hist_home_0, hist_away_0, data["odds_1x2"])
    f_home = score_info["facteur_home_2h"]
    f_away = score_info["facteur_away_2h"]

    joint_home, cop_home, pmfs_home = _build_pmf_par_equipe(
        hist_home, facteur_2h=f_home, demi_vie=demi_vie
    )
    joint_away, cop_away, pmfs_away = _build_pmf_par_equipe(
        hist_away, facteur_2h=f_away, demi_vie=demi_vie
    )

    mc1_pmf = product_pmf(joint_home)
    mc2_pmf = product_pmf(joint_away)

    # Dependance inter-equipes
    if rho_force is not None:
        rho = float(rho_force)
        methode_rho = "force"
        n_rho = 0
    else:
        rho, n_rho, methode_rho = estimer_correlation_inter_equipes(
            hist_home, hist_away
        )

    m1_vec = np.zeros(max(mc1_pmf.keys()) + 1)
    m2_vec = np.zeros(max(mc2_pmf.keys()) + 1)
    for k, v in mc1_pmf.items():
        m1_vec[k] = v
    for k, v in mc2_pmf.items():
        m2_vec[k] = v

    if abs(rho) >= 0.10:
        joint_mc, info_mc = joint_mc_via_copule(m1_vec, m2_vec, rho)
        global_pmf = global_depuis_joint_mc(joint_mc)
    else:
        global_pmf = convolve(mc1_pmf, mc2_pmf)
        info_mc = {
            "rho": rho, "tau": 0.0, "theta": 0.0,
            "independance_ok": True, "methode": "independance",
        }
    info_mc["n"] = n_rho
    info_mc["methode_estim"] = methode_rho

    stats = {
        "MC1": moments(mc1_pmf),
        "MC2": moments(mc2_pmf),
        "GLOBAL": moments(global_pmf),
        "p_zero_MC1": p_zero(mc1_pmf),
        "p_zero_MC2": p_zero(mc2_pmf),
        "p_zero_GLOBAL": p_zero(global_pmf),
    }

    pr = rapport_pagerank(hist_home, hist_away,
                          data["match"]["home_team"],
                          data["match"]["away_team"])

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

        p_over_val, p_under_val = p_over(pmf, ligne), p_under(pmf, ligne)

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

        rouges_home_5 = sum(m["cartons_rouges"] for m in hist_home_0[:5])
        rouges_away_5 = sum(m["cartons_rouges"] for m in hist_away_0[:5])
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
            "stable_sensibilite": True,  # provisoire, ajuste apres tornade
            "ic80_largeur": ev_over_info["ic80_largeur"],
            "contexte": ctx,
        }
        analyse_under = {
            "n": 15,
            "p_estime": p_under_val,
            "p_implicite_sans_marge": pi_under_prop,
            "ev_p20": ev_under_info["ev_p20"],
            "spread_modeles": abs(pi_under_prop - pi_under_pow),
            "stable_sensibilite": True,
            "ic80_largeur": ev_under_info["ic80_largeur"],
            "contexte": ctx,
        }
        res_over = evaluer_selection(analyse_over)
        res_under = evaluer_selection(analyse_under)

        marches[nom_marche] = {
            "ligne": ligne,
            "cotes": {
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
            },
            "marge_totale": marge,
            "ecart_methode_marge": abs(pi_over_prop - pi_over_pow),
        }

    zi_info = {
        "home_A1": pmfs_home["zero_inflation"]["A1"],
        "home_A2": pmfs_home["zero_inflation"]["A2"],
        "away_A1": pmfs_away["zero_inflation"]["A1"],
        "away_A2": pmfs_away["zero_inflation"]["A2"],
    }

    return {
        "match": data["match"],
        "stats": stats,
        "marches": marches,
        "pagerank": pr,
        "copule_home": cop_home,
        "copule_away": cop_away,
        "score_effect": score_info,
        "dependance_inter": info_mc,
        "zero_inflation": zi_info,
    }


def _decision(rapport, marche, cote):
    return "accept" if rapport["marches"][marche]["cotes"][cote]["gate"]["accepte"] else "reject"


def _recompute_verdict(rapport):
    sel = []
    for m, info in rapport["marches"].items():
        for c in COTES:
            if info["cotes"][c]["gate"]["accepte"]:
                sel.append(f"{m} {c.upper()} {info['ligne']}")
    rapport["selections_acceptees"] = sel
    rapport["verdict"] = (
        "SELECTION : " + " | ".join(sel) if sel
        else "AUCUNE SELECTION SUFFISAMMENT SOLIDE POUR CE MARCHE."
    )


def analyser_match(data):
    valider_match(data)

    base = _pipeline(data, {})

    # Decisions de base
    base_decisions = {
        (m, c): _decision(base, m, c) for m in MARCHES for c in COTES
    }

    # --- Tornade ---
    stabilite = {
        (m, c): {"stable": True, "flips": [], "scenarios": []}
        for m in MARCHES for c in COTES
    }

    for sc_nom, sc_params in SCENARIOS:
        try:
            r = _pipeline(data, sc_params)
        except Exception as e:
            continue
        for m in MARCHES:
            for c in COTES:
                new_dec = _decision(r, m, c)
                stabilite[(m, c)]["scenarios"].append({
                    "nom": sc_nom,
                    "decision": new_dec,
                    "p": float(r["marches"][m]["cotes"][c]["p"]),
                })
                if new_dec != base_decisions[(m, c)]:
                    stabilite[(m, c)]["stable"] = False
                    stabilite[(m, c)]["flips"].append(sc_nom)

    # --- Post-filtre : rejeter toute ligne instable ---
    for m in MARCHES:
        for c in COTES:
            st = stabilite[(m, c)]
            gate = base["marches"][m]["cotes"][c]["gate"]
            if not st["stable"]:
                if gate["accepte"]:
                    gate["accepte"] = False
                    gate["raisons_refus"].append(
                        "Critère 5 — instabilité en sensibilité "
                        "(flips : " + ", ".join(st["flips"]) + ")"
                    )
            else:
                # Marquer explicitement comme stable dans le gate
                if "Critère 5" not in " ".join(gate["raisons_refus"]):
                    gate["stable_sensibilite"] = True

    _recompute_verdict(base)
    base["stabilite"] = stabilite
    return base
