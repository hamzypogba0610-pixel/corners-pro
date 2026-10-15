"""Tests du modele hierarchique (avec ponderation temporelle)."""
import numpy as np
from core.hierarchical import (
    estimate_alpha_beta, posterior_alpha_beta, nb_pmf,
    build_pmf_equipe, build_joint, build_mc_global,
    poids_temporels, DEMI_VIE_DEFAUT,
)


def hist_avec_moyennes(a1, a2, date_debut=1):
    """Historique bidon, dates croissantes (ancien -> recent)."""
    return [
        {"date": f"2026-01-{(date_debut + i):02d}",
         "opponent": f"X{i}", "venue": "H",
         "corners_1h": a1[i], "corners_2h": a2[i],
         "resultat_final": "V", "resultat_mt": "N",
         "cartons_rouges": 0,
         "xg_pour": 1.5, "xg_contre": 0.9,
         "xgot_pour": 1.2, "xgot_contre": 0.7}
        for i in range(len(a1))
    ]


def test_poids_decroissants():
    """Le match le plus recent doit avoir le poids le plus eleve."""
    h = hist_avec_moyennes([3]*15, [4]*15)
    w = poids_temporels(h, demi_vie=8)
    assert len(w) == 15
    # Le dernier (plus recent) doit avoir le poids le plus fort
    assert w[-1] == max(w), f"poids le plus fort pas sur le plus recent : {w}"
    # Le premier (plus ancien) doit avoir le poids le plus faible
    assert w[0] == min(w)
    # Verif formule : w[0] = 2^(-14/8)
    attendu = 2.0 ** (-14 / 8)
    assert abs(w[0] - attendu) < 1e-9
    print(f"  OK : poids decroissants, w[0]={w[0]:.4f}, w[-1]={w[-1]:.4f}")


def test_poids_somme_finie():
    h = hist_avec_moyennes([3]*15, [4]*15)
    w = poids_temporels(h)
    assert np.sum(w) > 0
    assert np.all(w > 0)
    print(f"  OK : somme des poids = {np.sum(w):.3f}")


def test_alpha_beta_coherents():
    v = [2, 3, 4, 5, 3, 2, 4, 3, 5, 2]
    a, b = estimate_alpha_beta(v)
    assert a > 0 and b > 0
    mu = np.mean(v)
    mu_est = a / b
    assert abs(mu_est - mu) / mu < 0.5
    print(f"  OK : sans poids, alpha={a:.3f} beta={b:.3f}")


def test_alpha_beta_pondere():
    """Avec poids, la moyenne ponderee doit etre tiree vers les valeurs recentes."""
    v = [1, 1, 1, 1, 5, 5, 5, 5]  # 4 anciens bas, 4 recents hauts
    w = np.array([0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0])
    a, b = estimate_alpha_beta(v, poids=w)
    mu_pond = np.sum(w * np.array(v)) / np.sum(w)
    mu_est = a / b
    assert abs(mu_est - mu_pond) / mu_pond < 0.5
    print(f"  OK : moyenne ponderee {mu_pond:.3f}, mu_est {mu_est:.3f}")


def test_posterior_pondere():
    """Le posterior pondere doit donner plus d'importance aux recents."""
    v = [1, 1, 1, 1, 5, 5, 5, 5]
    w = np.array([0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0])
    a1, b1 = posterior_alpha_beta(v, 5.0, 2.0, poids=w)
    a2, b2 = posterior_alpha_beta(v, 5.0, 2.0)  # sans poids
    # Avec poids, la moyenne a posteriori (a/b) doit etre plus haute
    # car on donne plus de poids aux 5 recents
    assert a1 / b1 > a2 / b2
    print(f"  OK : posterior pondere tire vers les recents "
          f"({a1/b1:.3f} > {a2/b2:.3f})")


def test_shrinkage_vers_ligue():
    h = hist_avec_moyennes([5]*15, [5]*15)
    r = build_pmf_equipe(h)
    mu_a1 = np.sum(np.arange(len(r["A1"])) * r["A1"])
    assert 4.0 < mu_a1 < 6.0
    print(f"  OK : shrinkage coherence, mu_a1={mu_a1:.3f}")


def test_build_joint_dimensions():
    p1 = nb_pmf(20, 5.0, 2.0)
    p2 = nb_pmf(25, 5.0, 2.0)
    j = build_joint(p1, p2)
    assert j.shape == (21, 26)
    assert abs(j.sum() - 1.0) < 1e-9
    print("  OK : joint dimensions et somme")


def test_pipeline_complet():
    h_home = hist_avec_moyennes([3]*15, [4]*15)
    h_away = hist_avec_moyennes([2]*15, [3]*15)
    r = build_mc_global(h_home, h_away)
    for cle in ("mc1_pmf", "mc2_pmf", "global_pmf"):
        s = sum(r[cle].values())
        assert abs(s - 1.0) < 1e-9, f"{cle} somme={s}"
    print("  OK : pipeline complet")


def test_global_moyenne_coherente():
    h_home = hist_avec_moyennes([3]*15, [4]*15)
    h_away = hist_avec_moyennes([2]*15, [3]*15)
    r = build_mc_global(h_home, h_away)

    def moy(pmf):
        return sum(k * v for k, v in pmf.items())

    m1 = moy(r["mc1_pmf"]); m2 = moy(r["mc2_pmf"]); mg = moy(r["global_pmf"])
    assert abs(mg - (m1 + m2)) < 1e-9
    print(f"  OK : mg={mg:.3f} = m1({m1:.3f}) + m2({m2:.3f})")


def test_recent_form_impact():
    """Une equipe qui monte en forme doit avoir un MC moyen plus haut
    que la meme equipe avec les anciens et recents inverses."""
    # Forme montante : 2,2,2,2,2,2,2,2,4,4,4,4,4,4,4
    h_monte = hist_avec_moyennes(
        [2]*8 + [4]*7,
        [2]*8 + [4]*7
    )
    # Forme descendante : 4,4,4,4,4,4,4,2,2,2,2,2,2,2,2
    h_descend = hist_avec_moyennes(
        [4]*7 + [2]*8,
        [4]*7 + [2]*8
    )
    r_monte = build_mc_global(h_monte, h_descend)  # attention : h_descend en face
    # Comparer les MC moyens des deux equipes (monte vs descend)
    r1 = build_pmf_equipe(h_monte)
    r2 = build_pmf_equipe(h_descend)
    mu1 = np.sum(np.arange(len(r1["A1"])) * r1["A1"])
    mu2 = np.sum(np.arange(len(r2["A1"])) * r2["A1"])
    # h_monte (recents hauts) doit avoir un mu > h_descend (recents bas)
    assert mu1 > mu2, f"mu1={mu1} mu2={mu2}"
    print(f"  OK : forme recente capturee (mu_monte={mu1:.3f} > mu_desc={mu2:.3f})")


if __name__ == "__main__":
    tests = [
        test_poids_decroissants,
        test_poids_somme_finie,
        test_alpha_beta_coherents,
        test_alpha_beta_pondere,
        test_posterior_pondere,
        test_shrinkage_vers_ligue,
        test_build_joint_dimensions,
        test_pipeline_complet,
        test_global_moyenne_coherente,
        test_recent_form_impact,
    ]
    print("=== Tests du modele hierarchique (pondere) ===\n")
    r = 0
    for t in tests:
        try:
            t(); r += 1
        except Exception as e:
            print(f"  ECHEC : {t.__name__} -> {e}")
    print(f"\n=== {r}/{len(tests)} tests reussis ===")
    if r != len(tests):
        raise SystemExit(1)
