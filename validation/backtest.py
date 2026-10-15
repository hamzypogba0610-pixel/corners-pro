"""
Backtest glissant (rolling origin).
Reference : MULTICORNER 360 v2, section 6.
"""
import numpy as np


def construire_dataset_synthetique(n=80, seed=42):
    """
    Dataset de matchs passe : 4 equipes qui jouent souvent entre elles,
    pour garantir un historique suffisant.
    """
    rng = np.random.default_rng(seed)
    equipes = ["Paris", "Lyon", "Marseille", "Monaco"]
    force = {eq: rng.uniform(0.8, 1.4) for eq in equipes}
    matchs = []
    for i in range(n):
        home = equipes[i % 4]
        away = equipes[(i + 1 + (i // 4) % 3) % 4]
        if home == away:
            away = equipes[(i + 2) % 4]
        base_1h_h = 2.5 * force[home] * rng.uniform(0.7, 1.3)
        base_2h_h = 3.0 * force[home] * rng.uniform(0.7, 1.3)
        base_1h_a = 2.0 * force[away] * rng.uniform(0.7, 1.3)
        base_2h_a = 2.5 * force[away] * rng.uniform(0.7, 1.3)
        c1h_h = max(0, int(round(rng.poisson(base_1h_h))))
        c2h_h = max(0, int(round(rng.poisson(base_2h_h))))
        c1h_a = max(0, int(round(rng.poisson(base_1h_a))))
        c2h_a = max(0, int(round(rng.poisson(base_2h_a))))
        matchs.append({
            "date": f"2026-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
            "home": home, "away": away,
            "c1h_home": c1h_h, "c2h_home": c2h_h,
            "c1h_away": c1h_a, "c2h_away": c2h_a,
        })
    return matchs


def _hist_depuis_dataset(dataset, jusqu_a, equipe, venue):
    """Historique d'une equipe avec uniquement les matchs STRICTEMENT anterieurs."""
    hist = []
    for i in range(jusqu_a - 1, -1, -1):
        m = dataset[i]
        if venue == "H" and m["home"] == equipe:
            hist.append({
                "date": m["date"], "opponent": m["away"], "venue": "H",
                "corners_1h": m["c1h_home"], "corners_2h": m["c2h_home"],
                "resultat_final": "V", "resultat_mt": "N",
                "cartons_rouges": 0,
                "xg_pour": 1.5, "xg_contre": 1.0,
                "xgot_pour": 1.2, "xgot_contre": 0.8,
            })
        elif venue == "A" and m["away"] == equipe:
            hist.append({
                "date": m["date"], "opponent": m["home"], "venue": "A",
                "corners_1h": m["c1h_away"], "corners_2h": m["c2h_away"],
                "resultat_final": "D", "resultat_mt": "N",
                "cartons_rouges": 0,
                "xg_pour": 1.2, "xg_contre": 1.5,
                "xgot_pour": 1.0, "xgot_contre": 1.2,
            })
        if len(hist) >= 15:
            break
    return hist


def _p_over_depuis_pmf(pmf, ligne):
    return float(sum(p for m, p in pmf.items() if m > ligne))


def brier_score(p_hat, y_reel):
    return float((p_hat - y_reel) ** 2)


def log_loss(p_hat, y_reel, eps=1e-15):
    p = max(eps, min(1 - eps, p_hat))
    return float(-(y_reel * np.log(p) + (1 - y_reel) * np.log(1 - p)))


def rps_discret(pmf_dict, valeur_reelle, k_max):
    ks = list(range(k_max + 1))
    cdf_pred = np.cumsum([pmf_dict.get(k, 0.0) for k in ks])
    cdf_obs = np.array([1.0 if k >= valeur_reelle else 0.0 for k in ks])
    return float(np.sum((cdf_pred - cdf_obs) ** 2) / k_max) if k_max > 0 else 0.0


def executer_backtest(dataset, ligne=12.5, k_min=25, marche="MC1",
                      n_min_hist=5):
    """
    Pour chaque match k >= k_min, on predit en utilisant uniquement les
    matchs strictement anterieurs.
    """
    from core.analysis import _build_pmf_par_equipe
    from core.product_law import product_pmf, convolve

    predictions = []
    erreurs_hist = 0
    erreurs_calcul = 0

    for k in range(k_min, len(dataset)):
        m = dataset[k]
        hist_home = _hist_depuis_dataset(dataset, k, m["home"], "H")
        hist_away = _hist_depuis_dataset(dataset, k, m["away"], "A")
        if len(hist_home) < n_min_hist or len(hist_away) < n_min_hist:
            erreurs_hist += 1
            continue
        try:
            joint_home, _, _ = _build_pmf_par_equipe(hist_home)
            joint_away, _, _ = _build_pmf_par_equipe(hist_away)
            mc1_pmf = product_pmf(joint_home)
            mc2_pmf = product_pmf(joint_away)
        except Exception:
            erreurs_calcul += 1
            continue

        mc1_reel = m["c1h_home"] * m["c2h_home"]
        mc2_reel = m["c1h_away"] * m["c2h_away"]

        if marche == "MC1":
            pmf = mc1_pmf
            reel = mc1_reel
        elif marche == "MC2":
            pmf = mc2_pmf
            reel = mc2_reel
        else:
            pmf = convolve(mc1_pmf, mc2_pmf)
            reel = mc1_reel + mc2_reel

        p_over = _p_over_depuis_pmf(pmf, ligne)
        y_reel = 1.0 if reel > ligne else 0.0
        predictions.append({
            "k": k, "p_hat": p_over, "y": y_reel,
            "reel": reel, "pmf": pmf,
        })

    if not predictions:
        return {
            "n_pred": 0, "n_erreurs_hist": erreurs_hist,
            "n_erreurs_calcul": erreurs_calcul,
            "brier": None, "log_loss": None, "rps": None,
            "pit": [], "predictions": [],
        }

    briers = [brier_score(p["p_hat"], p["y"]) for p in predictions]
    lls = [log_loss(p["p_hat"], p["y"]) for p in predictions]

    k_max_rps = max(max(p["pmf"].keys()) for p in predictions)
    k_max_rps = min(k_max_rps, 60)
    rpss = [rps_discret(p["pmf"], p["reel"], k_max_rps) for p in predictions]

    pits = []
    for p in predictions:
        ks = sorted(p["pmf"].keys())
        cum = 0.0
        for kk in ks:
            cum += p["pmf"][kk]
            if kk >= p["reel"]:
                pits.append(cum)
                break

    return {
        "n_pred": len(predictions),
        "n_erreurs_hist": erreurs_hist,
        "n_erreurs_calcul": erreurs_calcul,
        "brier": float(np.mean(briers)),
        "log_loss": float(np.mean(lls)),
        "rps": float(np.mean(rpss)),
        "pit": [float(x) for x in pits],
        "brier_baseline": float(np.mean([
            (0.5 - p["y"]) ** 2 for p in predictions
        ])),
        "predictions": [
            {"k": p["k"], "p_hat": p["p_hat"], "y": p["y"], "reel": p["reel"]}
            for p in predictions
        ],
    }


def histogramme_pit(pit_values, n_bins=10):
    """Histogramme PIT (uniforme = bien calibre)."""
    if pit_values is None or len(pit_values) == 0:
        return {"bins": [], "counts": [], "n": 0}
    arr = np.asarray(pit_values, dtype=float)
    counts, edges = np.histogram(arr, bins=n_bins, range=(0.0, 1.0))
    return {
        "bins": [float(e) for e in edges],
        "counts": [int(c) for c in counts],
        "n": int(len(pit_values)),
        }
