"""
Serveur web du bot corners-pro.
Version 13.1 : correction _fmt_pct + traces de debug.
"""
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
import html as html_mod

from data.schema import ValidationError
from core.analysis import parser_historique, analyser_match
from validation.backtest import (
    construire_dataset_synthetique, executer_backtest, histogramme_pit,
)


app = FastAPI(title="corners-pro", version="1.3.1")

if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


def _page(titre, corps, scripts=""):
    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{titre}</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ font-family: -apple-system, sans-serif; background:#0d1117; color:#c9d1d9;
           margin:0; padding:16px; line-height:1.5; }}
    h1 {{ color:#58a6ff; font-size:22px; }}
    h2 {{ color:#79c0ff; font-size:17px; margin-top:24px;
         border-bottom:1px solid #21262d; padding-bottom:6px; }}
    h3 {{ color:#a5d6ff; font-size:15px; margin-top:18px; }}
    a {{ color:#58a6ff; }}
    label {{ display:block; margin:12px 0 4px; color:#8b949e; font-size:13px; }}
    input, select, textarea {{
      width:100%; padding:10px; border-radius:8px;
      border:1px solid #30363d; background:#161b22; color:#c9d1d9;
      font-family:inherit; font-size:15px;
    }}
    textarea {{ min-height:220px; font-family:monospace; font-size:12px; }}
    button {{ margin-top:14px; width:100%; padding:14px;
      background:#238636; color:white; border:none;
      border-radius:8px; font-size:16px; font-weight:bold; }}
    button:active {{ background:#2ea043; }}
    .btn-sec {{ background:#21262d; color:#c9d1d9; padding:8px 12px;
                font-size:13px; width:auto; margin:4px 4px 0 0;
                font-weight:normal; }}
    .btn-sec:active {{ background:#30363d; }}
    .card {{ background:#161b22; border:1px solid #30363d; border-radius:10px;
             padding:14px; margin-top:12px; }}
    .ok {{ color:#3fb950; font-weight:bold; }}
    .warn {{ color:#d29922; font-weight:bold; }}
    .err {{ color:#f85149; font-weight:bold; }}
    .info {{ color:#8b949e; font-size:13px; margin:6px 0; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; margin-top:8px; }}
    th, td {{ padding:8px; text-align:left; border-bottom:1px solid #21262d; }}
    th {{ color:#8b949e; font-weight:normal; }}
    .verdict {{ padding:16px; border-radius:10px; margin-top:16px;
                text-align:center; font-size:16px; font-weight:bold; }}
    .verdict-oui {{ background:#238636; color:white; }}
    .verdict-non {{ background:#30363d; color:#c9d1d9; }}
    .raisons {{ background:#21262d; padding:10px; border-radius:6px;
                margin-top:8px; font-size:12px; color:#d29922; }}
    .raisons ul {{ margin:6px 0 0 16px; padding:0; }}
    code {{ background:#21262d; padding:2px 6px; border-radius:4px; font-size:11px; }}
    .barre-equipe {{ margin-top:6px; display:flex; flex-wrap:wrap; gap:6px; align-items:center; }}
    .barre-equipe select {{ flex:1; min-width:150px; font-size:13px; padding:8px; }}
    .ic80 {{ font-size:11px; color:#8b949e; }}
    .encadre {{ background:#1f2937; border-left:3px solid #58a6ff;
                padding:8px 12px; margin:8px 0; font-size:12px; color:#c9d1d9; }}
    .badge {{ display:inline-block; padding:3px 8px; border-radius:12px;
              font-size:11px; font-weight:bold; }}
    .badge-off {{ background:#1f6feb; color:white; }}
    .badge-def {{ background:#8b1f1f; color:white; }}
    .badge-eq {{ background:#30363d; color:#c9d1d9; }}
  </style>
</head>
<body>
{corps}
{scripts}
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def racine():
    return _page("corners-pro", """
      <h1>corners-pro</h1>
      <div class="card">
        <span class="ok">🟢 Bot en ligne</span>
        <div class="info">Version 1.3.1 — bug corrigé</div>
      </div>
      <p style="margin-top:24px;"><a href="/nouveau">→ Lancer une nouvelle analyse</a></p>
      <p style="margin-top:8px;"><a href="/equipes">→ Mes équipes sauvegardées</a></p>
      <p style="margin-top:8px;"><a href="/backtest">→ Lancer un backtest</a></p>
      <p style="margin-top:8px;"><a href="/suivi">→ Mon suivi de prédictions</a></p>
    """)


@app.get("/nouveau", response_class=HTMLResponse)
def nouveau():
    return _page("Nouvelle analyse", """
      <h1>Nouvelle analyse</h1>
      <form method="post" action="/analyser">

        <h2>1. Match</h2>
        <label>Compétition</label>
        <input name="competition" required placeholder="Ligue 1">
        <label>Équipe domicile</label>
        <input name="home_team" id="home_team" required placeholder="Paris">
        <label>Équipe extérieur</label>
        <input name="away_team" id="away_team" required placeholder="Lyon">
        <label>Date</label>
        <input name="date" required placeholder="2026-02-01">
        <label>Mode</label>
        <select name="mode">
          <option value="RAPIDE">RAPIDE (15 s)</option>
          <option value="COMPLET">COMPLET (45 s)</option>
        </select>

        <h2>2. Cotes 1X2 du match</h2>
        <div class="info">Utilisées pour pondérer la 2e mi-temps (effet score).</div>
        <label>Cote Domicile</label>
        <input name="odd_home" type="number" step="0.01" min="1.01" required placeholder="2.10">
        <label>Cote Nul</label>
        <input name="odd_draw" type="number" step="0.01" min="1.01" required placeholder="3.40">
        <label>Cote Extérieur</label>
        <input name="odd_away" type="number" step="0.01" min="1.01" required placeholder="3.20">

        <h2>3. Historique équipe domicile</h2>
        <div class="encadre">
          <b>📋 Règle de saisie :</b><br>
          • <b>10 premiers matchs</b> = matchs joués <b>à domicile</b><br>
          • <b>5 derniers matchs</b> = matchs <b>globaux</b>
        </div>
        <div class="info">
          <code>date,adversaire,c1h,c2h,rf,rmt,rouge,xg,xga,xgot,xgota</code>
        </div>
        <div class="barre-equipe">
          <select id="select_home">
            <option value="">— Choisir une équipe —</option>
          </select>
          <button type="button" class="btn-sec"
                  onclick="remplirDepuisListe('home_history','select_home')">
            📂 Charger
          </button>
          <button type="button" class="btn-sec"
                  onclick="sauvegarderDepuisTextarea('home_history','home_team')">
            💾 Sauvegarder
          </button>
        </div>
        <textarea name="home_history" id="home_history" required
                  placeholder="2026-01-15,Marseille,3,4,V,N,0,1.5,0.9,1.2,0.7"></textarea>

        <h2>4. Historique équipe extérieur</h2>
        <div class="encadre">
          <b>📋 Règle de saisie :</b><br>
          • <b>10 premiers matchs</b> = matchs joués <b>à l'extérieur</b><br>
          • <b>5 derniers matchs</b> = matchs <b>globaux</b>
        </div>
        <div class="barre-equipe">
          <select id="select_away">
            <option value="">— Choisir une équipe —</option>
          </select>
          <button type="button" class="btn-sec"
                  onclick="remplirDepuisListe('away_history','select_away')">
            📂 Charger
          </button>
          <button type="button" class="btn-sec"
                  onclick="sauvegarderDepuisTextarea('away_history','away_team')">
            💾 Sauvegarder
          </button>
        </div>
        <textarea name="away_history" id="away_history" required
                  placeholder="2026-01-15,Nice,2,2,N,V,0,1.0,1.2,0.8,0.9"></textarea>

        <h2>5. Cotes des marchés</h2>
        <h3>MC1</h3>
        <label>Ligne (ex 12.5)</label>
        <input name="mc1_line" type="number" step="0.5" required placeholder="12.5">
        <label>Cote Over</label>
        <input name="mc1_over" type="number" step="0.01" min="1.01" required placeholder="1.90">
        <label>Cote Under</label>
        <input name="mc1_under" type="number" step="0.01" min="1.01" required placeholder="1.90">

        <h3>MC2</h3>
        <label>Ligne</label>
        <input name="mc2_line" type="number" step="0.5" required placeholder="12.5">
        <label>Cote Over</label>
        <input name="mc2_over" type="number" step="0.01" min="1.01" required placeholder="1.85">
        <label>Cote Under</label>
        <input name="mc2_under" type="number" step="0.01" min="1.01" required placeholder="1.95">

        <h3>GLOBAL</h3>
        <label>Ligne</label>
        <input name="glo_line" type="number" step="0.5" required placeholder="25.5">
        <label>Cote Over</label>
        <input name="glo_over" type="number" step="0.01" min="1.01" required placeholder="1.88">
        <label>Cote Under</label>
        <input name="glo_under" type="number" step="0.01" min="1.01" required placeholder="1.92">

        <button type="submit">Analyser le match</button>
      </form>
    """, scripts="""
      <script src="/static/equipes.js"></script>
      <script>
        document.addEventListener("DOMContentLoaded", function() {
          peuplerSelect("select_home", "H");
          peuplerSelect("select_away", "A");
        });
      </script>
    """)


@app.get("/equipes", response_class=HTMLResponse)
def equipes():
    return _page("Mes équipes", """
      <h1>Mes équipes</h1>
      <div class="card">
        <div class="info">Ces équipes sont stockées dans <b>ce navigateur</b>.</div>
        <button type="button" class="btn-sec" onclick="exporterTout()">⬇️ Exporter tout</button>
        <label style="margin-top:12px;">Importer un fichier JSON</label>
        <input type="file" accept=".json" onchange="importerFichier(this)">
      </div>
      <div id="liste" style="margin-top:16px;"></div>
      <p style="margin-top:20px;"><a href="/nouveau">← Nouvelle analyse</a></p>
    """, scripts="""
      <script src="/static/equipes.js"></script>
      <script>
        document.addEventListener("DOMContentLoaded", function() {
          const equipes = lireEquipes();
          const noms = Object.keys(equipes).sort();
          const cible = document.getElementById("liste");
          if (noms.length === 0) {
            cible.innerHTML = '<p class="info">Aucune équipe sauvegardée pour le moment.</p>';
            return;
          }
          let html = '<table><tr><th>Équipe</th><th>Lieu</th><th></th></tr>';
          noms.forEach(n => {
            const e = equipes[n];
            html += '<tr><td>' + n + '</td><td>' + e.venue + '</td>'
                 + '<td><button type="button" class="btn-sec" '
                 + 'onclick="if(supprimerEquipe(\\'' + n.replace(/'/g, "\\\\'") + '\\')) location.reload();">'
                 + '🗑️</button></td></tr>';
          });
          html += '</table>';
          cible.innerHTML = html;
        });
      </script>
    """)


@app.get("/backtest", response_class=HTMLResponse)
def page_backtest():
    try:
        ds = construire_dataset_synthetique(n=80, seed=42)
        r_mc1 = executer_backtest(ds, ligne=12.5, k_min=25, marche="MC1")
        r_mc2 = executer_backtest(ds, ligne=12.5, k_min=25, marche="MC2")
        r_glo = executer_backtest(ds, ligne=25.5, k_min=25, marche="GLOBAL")
    except Exception as e:
        return _page("Backtest — erreur", f"""
          <h1 class="err">Erreur backtest</h1>
          <div class="card"><pre>{type(e).__name__} : {e}</pre></div>
          <p><a href="/">← Retour</a></p>
        """)

    def _bloc(nom, r):
        if r["n_pred"] == 0:
            return f"<tr><td>{nom}</td><td colspan='5'>Pas de prédictions</td></tr>"
        brier_base = r["brier_baseline"]
        brier = r["brier"]
        gain = (brier_base - brier) / brier_base * 100 if brier_base else 0
        couleur = "ok" if gain > 0 else "err"
        return f"""
          <tr>
            <td>{nom}</td>
            <td>{r['n_pred']}</td>
            <td>{brier:.4f}</td>
            <td>{brier_base:.4f}</td>
            <td class="{couleur}">{gain:+.1f}%</td>
            <td>{r['rps']:.4f}</td>
          </tr>
        """

    h_pit = histogramme_pit(r_mc1["pit"], n_bins=10)
    hist_html = _rendre_histogramme_pit(h_pit)

    return _page("Backtest", f"""
      <h1>Backtest glissant</h1>
      <p class="info">80 matchs simulés — rolling origin, sans fuite.</p>
      <h2>Métriques principales</h2>
      <table>
        <tr><th>Marché</th><th>n</th><th>Brier</th><th>Brier baseline</th>
            <th>Gain</th><th>RPS</th></tr>
        {_bloc("MC1 (12.5)", r_mc1)}
        {_bloc("MC2 (12.5)", r_mc2)}
        {_bloc("GLOBAL (25.5)", r_glo)}
      </table>
      <h2>Histogramme PIT (MC1)</h2>
      {hist_html}
      <p style="margin-top:24px;"><a href="/">← Accueil</a></p>
    """)


@app.get("/suivi", response_class=HTMLResponse)
def page_suivi():
    return _page("Mon suivi", """
      <h1>Mon suivi de prédictions</h1>
      <div class="card">
        <button type="button" class="btn-sec" onclick="exporterSuivi()">⬇️ Exporter</button>
        <label style="margin-top:12px;">Importer un fichier JSON</label>
        <input type="file" accept=".json" onchange="importerSuivi(this)">
      </div>
      <div id="stats"></div>
      <div id="liste" style="margin-top:16px;"></div>
      <p style="margin-top:20px;"><a href="/">← Accueil</a></p>
    """, scripts="""
      <script src="/static/suivi.js"></script>
      <script>
        document.addEventListener("DOMContentLoaded", function() {
          const stats = calculerStatsSuivi();
          const statsEl = document.getElementById("stats");
          if (stats.n > 0) {
            const pnlColor = stats.pnl > 0 ? "#3fb950" : "#f85149";
            statsEl.innerHTML = `
              <h2>Performance réelle</h2>
              <table>
                <tr><th>Prédictions terminées</th><td>${stats.n}</td></tr>
                <tr><th>Brier score</th><td>${stats.brier.toFixed(4)}</td></tr>
                <tr><th>EV prévu moyen</th><td>${stats.ev_prevu_moyen.toFixed(4)}</td></tr>
                <tr><th>EV réalisé moyen</th><td>${stats.ev_reel_moyen.toFixed(4)}</td></tr>
                <tr><th>Taux de réussite</th><td>${stats.taux_reussite.toFixed(1)}%</td></tr>
                <tr><th>ROI</th><td style="color:${pnlColor};"><b>${stats.roi.toFixed(2)}%</b></td></tr>
                <tr><th>P&L</th><td style="color:${pnlColor};">${stats.pnl.toFixed(2)}</td></tr>
              </table>
            `;
          } else {
            statsEl.innerHTML = '<p class="info">Aucune prédiction terminée.</p>';
          }
          const preds = lireSuivi();
          const liste = document.getElementById("liste");
          if (preds.length === 0) {
            liste.innerHTML = '<p class="info">Aucune prédiction enregistrée.</p>';
            return;
          }
          let html = '<h2>Historique</h2>';
          for (const p of preds) {
            const res = p.resultat_reel || "—";
            let badge = "";
            if (res === "gagne") badge = '<span class="ok">✓ GAGNÉ</span>';
            else if (res === "perdu") badge = '<span class="err">✗ PERDU</span>';
            else if (res === "push") badge = '<span class="warn">push</span>';
            else badge = '<span class="info">en attente</span>';
            const dateShort = (p.date_enregistrement || "").slice(0, 10);
            html += `
              <div class="card">
                <b>${p.match}</b><br>
                <span class="info">${p.marche} ${p.pari} — ligne ${p.ligne} — cote ${p.cote}</span><br>
                <span class="info">p : ${(p.p_estime*100).toFixed(2)}% — EV : ${p.ev_prevu.toFixed(4)}</span><br>
                <span class="info">${dateShort}</span> ${badge}
                <div style="margin-top:8px;">
                  <button type="button" class="btn-sec" onclick="setResultat('${p.id}', 'gagne')">Gagné</button>
                  <button type="button" class="btn-sec" onclick="setResultat('${p.id}', 'perdu')">Perdu</button>
                  <button type="button" class="btn-sec" onclick="setResultat('${p.id}', 'push')">Push</button>
                  <button type="button" class="btn-sec" onclick="if(supprimerPrediction('${p.id}')) location.reload();">🗑️</button>
                </div>
              </div>
            `;
          }
          liste.innerHTML = html;
        });
      </script>
    """)

    def _fmt_pct(x):
        return f"{x*100:.2f}%"


def _fmt_ev(x):
    return f"{x:+.4f}"


def _rendre_histogramme_pit(h):
    if not h or h["n"] == 0:
        return '<p class="info">Pas de données PIT.</p>'
    counts = h["counts"]
    cmax = max(counts) if counts else 1
    barres = "".join(
        f'<div style="flex:1; background:#58a6ff;'
        f' height:{int(80 * c / cmax)}px;'
        f' border-radius:2px 2px 0 0; min-height:2px;"'
        f' title="{c}"></div>'
        for c in counts
    )
    return f"""
      <div style="display:flex; align-items:flex-end; gap:4px;
                  height:90px; padding:6px; background:#21262d;
                  border-radius:6px; margin-top:8px;">
        {barres}
      </div>
      <p class="info">n = {h['n']}. Barre plate = bonne calibration.</p>
    """


def _badge_style(label):
    if label == "offensif":
        return '<span class="badge badge-off">⚔️ OFFENSIF</span>'
    if label == "defensif":
        return '<span class="badge badge-def">🛡️ DÉFENSIF</span>'
    if label == "equilibre":
        return '<span class="badge badge-eq">⚖️ ÉQUILIBRÉ</span>'
    return '<span class="badge badge-eq">? inconnu</span>'


def _rendre_ligue_style(ligue, style_home, style_away, nom_home, nom_away):
    if not ligue:
        return ""
    connue = ligue.get("connue", False)
    if connue:
        statut_ligue = (
            f'<span class="ok">détectée : {ligue.get("ligue_detectee")}</span>'
        )
    else:
        statut_ligue = (
            '<span class="warn">non reconnue — référence globale</span>'
        )
    taux = ligue.get("taux_reference", 10.0)
    ref = ligue.get("reference_globale", 10.0)
    ratio = ligue.get("ratio_brut", 1.0)
    exposant = ligue.get("exposant", 0.3)
    facteur = ligue.get("facteur", 1.0)
    return f"""
      <h2>Contexte championnat et style d'équipe</h2>
      <p class="info">
        Compétition : <b>{html_mod.escape(str(ligue.get("competition", "")))}</b>
        — {statut_ligue}<br>
        Taux de référence : <b>{taux:.2f}</b> corners/match
        (référence : {ref:.1f})<br>
        Ratio brut : {ratio:.3f} → exposant {exposant} →
        <b>facteur de ligue = {facteur:.4f}</b>
      </p>
      <p class="info" style="font-size:11px;">
        ⚠️ Table approximative des championnats.
      </p>
      <h3>Style des équipes</h3>
      <table>
        <tr><th>Équipe</th><th>Style</th><th>Moy. équipe</th>
            <th>Moy. attendue</th><th>Ratio</th></tr>
        <tr>
          <td>{html_mod.escape(nom_home)} (dom.)</td>
          <td>{_badge_style(style_home.get("label"))}</td>
          <td>{style_home.get("moyenne_equipe", 0):.2f}</td>
          <td>{style_home.get("moyenne_attendue", 0):.2f}</td>
          <td>{style_home.get("ratio", 1):.3f}</td>
        </tr>
        <tr>
          <td>{html_mod.escape(nom_away)} (ext.)</td>
          <td>{_badge_style(style_away.get("label"))}</td>
          <td>{style_away.get("moyenne_equipe", 0):.2f}</td>
          <td>{style_away.get("moyenne_attendue", 0):.2f}</td>
          <td>{style_away.get("ratio", 1):.3f}</td>
        </tr>
      </table>
    """


def _rendre_integrite_calibration(integrite, cal_active, nb_marches):
    if not integrite:
        return ""
    if integrite.get("ok"):
        statut_int = '<span class="ok">✓ Intégrité validée</span>'
    else:
        statut_int = '<span class="err">✗ Intégrité compromise</span>'
    bloquants = integrite.get("problemes_bloquants", [])
    avertissements = integrite.get("avertissements", [])
    bloc_bloq = ""
    if bloquants:
        items = "".join(
            f"<li>{html_mod.escape(str(x))}</li>" for x in bloquants[:10]
        )
        bloc_bloq = f'<div class="raisons">Bloquants<ul>{items}</ul></div>'
    bloc_av = ""
    if avertissements:
        items = "".join(
            f"<li>{html_mod.escape(str(x))}</li>" for x in avertissements[:10]
        )
        bloc_av = f'<div class="raisons">Avertissements<ul>{items}</ul></div>'
    if cal_active:
        statut_cal = '<span class="ok">✓ Calibration active</span>'
    else:
        statut_cal = (
            '<span class="info">Calibration inactive — '
            'nécessite 50+ prédictions</span>'
        )
    return f"""
      <h2>Couche d'intégrité et calibration</h2>
      <p class="info">
        <b>Intégrité</b> : {statut_int} — {nb_marches} marchés vérifiés<br>
        <b>Calibration</b> : {statut_cal}
      </p>
      {bloc_bloq}
      {bloc_av}
    """


def _rendre_ligne_pari(nom, cote_nom, c, match_label):
    label = cote_nom.upper()
    gate = c["gate"]
    couleur = "ok" if gate["accepte"] else "err"
    statut = "✅ RETENU" if gate["accepte"] else "❌ REFUSÉ"
    ev_info = c["ev_info"]
    ic_bas = ev_info["ic80_bas"]
    ic_haut = ev_info["ic80_haut"]
    raisons_html = ""
    if not gate["accepte"] and gate["raisons_refus"]:
        items = "".join(f"<li>{r}</li>" for r in gate["raisons_refus"])
        raisons_html = f'<div class="raisons">Pourquoi ce refus ?<ul>{items}</ul></div>'
    m_esc = html_mod.escape(match_label, quote=True)
    nom_esc = html_mod.escape(nom, quote=True)
    label_esc = html_mod.escape(label, quote=True)
    bouton_html = ""
    if gate["accepte"]:
        bouton_html = (
            f'<button type="button" class="btn-sec" '
            f'onclick="enregistrerDepuisRapport(this)" '
            f'data-match="{m_esc}" '
            f'data-marche="{nom_esc}" '
            f'data-pari="{label_esc}" '
            f'data-cote="{c["cote"]}" '
            f'data-p="{c["p"]}" '
            f'data-ev="{ev_info["ev_mediane"]}" '
            f'data-ligne="0">'
            f'📌 Enregistrer cette prédiction</button>'
        )
    p_brut = c.get("p_brut", c["p"])
    calibree = c.get("calibree", False)
    ligne_p = f"{_fmt_pct(c['p'])}"
    if calibree and abs(p_brut - c["p"]) > 1e-6:
        ligne_p += f'<br><span class="ic80">brut {p_brut*100:.2f}%</span>'
    return f"""
      <tr>
        <td><b>{label}</b></td>
        <td>{ligne_p}<br>
            <span class="ic80">[{ic_bas*100:.1f} – {ic_haut*100:.1f}]</span>
        </td>
        <td>{_fmt_pct(c['p_impl_prop'])}</td>
        <td>{c['cote']:.2f}</td>
        <td>{_fmt_ev(ev_info['ev_mediane'])} / {_fmt_ev(ev_info['ev_p20'])}</td>
        <td>{c['kelly']*100:.2f}%</td>
        <td class="{couleur}">{statut}</td>
      </tr>
      <tr><td colspan="7" style="padding:0; border:none;">{raisons_html}</td></tr>
      <tr><td colspan="7" style="padding:4px 0 12px 0; border:none;">{bouton_html}</td></tr>
    """


def _rendre_marche(nom, info, match_label):
    rows = (
        _rendre_ligne_pari(nom, "over", info["cotes"]["over"], match_label)
        + _rendre_ligne_pari(nom, "under", info["cotes"]["under"], match_label)
    )
    return f"""
      <h3>{nom} — ligne {info['ligne']}</h3>
      <p class="info">Marge bookmaker : {_fmt_pct(info['marge_totale'])}
         — écart méthodes : {_fmt_pct(info['ecart_methode_marge'])}</p>
      <table>
        <tr><th>Pari</th><th>p estimée</th><th>p implicite</th>
            <th>Cote</th><th>EV méd / p20</th><th>Kelly</th><th>Statut</th></tr>
        {rows}
      </table>
    """


def _rendre_effet_score(se):
    if not se:
        return ""
    pf = se["probs_ft"]
    hh = se["ht_home"]
    ha = se["ht_away"]
    ratios = se["ratios"]
    sr = se.get("stats_ratios", {})

    def _r(s):
        if s in sr:
            return f"{ratios[s]:.3f} (n={sr[s]['n']})"
        return f"{ratios[s]:.3f}"

    return f"""
      <h2>Effet score (pondération 2e MT)</h2>
      <p class="info">
        Cotes 1X2 → probas fin de match (marge {_fmt_pct(pf['marge'])}) :<br>
        Domicile {_fmt_pct(pf['p_home'])} | Nul {_fmt_pct(pf['p_draw'])} | Extérieur {_fmt_pct(pf['p_away'])}
      </p>
      <table>
        <tr><th>Équipe</th><th>Mène HT</th><th>Nul HT</th><th>Menée HT</th><th>Facteur 2e MT</th></tr>
        <tr><td>Domicile</td>
            <td>{_fmt_pct(hh['p_leads_ht'])}</td>
            <td>{_fmt_pct(hh['p_draw_ht'])}</td>
            <td>{_fmt_pct(hh['p_trails_ht'])}</td>
            <td><b>{se['facteur_home_2h']:.3f}</b></td></tr>
        <tr><td>Extérieur</td>
            <td>{_fmt_pct(ha['p_leads_ht'])}</td>
            <td>{_fmt_pct(ha['p_draw_ht'])}</td>
            <td>{_fmt_pct(ha['p_trails_ht'])}</td>
            <td><b>{se['facteur_away_2h']:.3f}</b></td></tr>
      </table>
      <p class="info">
        Ratios empiriques (2e MT selon scénario HT) :
        mène {_r('V')} | nul {_r('N')} | menée {_r('D')}
      </p>
    """


def _rendre_dependance_inter(dep):
    if not dep:
        return ""
    rho = dep.get("rho", 0.0)
    tau = dep.get("tau", 0.0)
    theta = dep.get("theta", 0.0)
    n = dep.get("n", 0)
    indep = dep.get("independance_ok", True)
    methode = dep.get("methode_estim", "")
    if indep:
        statut = '<span class="info">indépendance retenue</span>'
    else:
        statut = '<span class="warn">dépendance détectée</span>'
    return f"""
      <h2>Dépendance inter-équipes (MC1 ↔ MC2)</h2>
      <p class="info">
        Proxy : corr(corners totaux, xG concédés).<br>
        Estimation sur {n} matchs — méthode : {methode}<br>
        <b>ρ</b> = {rho:.3f} | <b>τ</b> = {tau:.3f} | <b>θ</b> = {theta:.3f}<br>
        Seuil : |ρ| ≥ 0.10 — {statut}
      </p>
    """


def _rendre_zero_inflation(zi):
    if not zi:
        return ""

    def _bloc(label, info):
        if not info:
            return ""
        inflation = info.get("inflation", False)
        statut = (
            '<span class="warn">inflation détectée</span>' if inflation
            else '<span class="info">pas d\'inflation</span>'
        )
        return f"""
          <tr>
            <td>{label}</td>
            <td>{info.get('p_zero_obs', 0.0)*100:.1f}%</td>
            <td>{info.get('p_zero_nb', 0.0)*100:.1f}%</td>
            <td>{info.get('ecart', 0.0)*100:+.2f} pts</td>
            <td>{info.get('pi', 0.0):.3f}</td>
            <td>{statut}</td>
          </tr>
        """

    return f"""
      <h2>Inflation de zéros</h2>
      <p class="info">Test : P(0) observé vs P(0) attendu sous NB.</p>
      <table>
        <tr><th>Ligne</th><th>P(0) observé</th><th>P(0) NB</th>
            <th>Écart</th><th>π (ajusté)</th><th>Statut</th></tr>
        {_bloc("Domicile — 1re MT", zi.get("home_A1"))}
        {_bloc("Domicile — 2e MT", zi.get("home_A2"))}
        {_bloc("Extérieur — 1re MT", zi.get("away_A1"))}
        {_bloc("Extérieur — 2e MT", zi.get("away_A2"))}
      </table>
    """


def _rendre_tornade(stabilite, mode_utilise="COMPLET"):
    if not stabilite:
        return ""
    if mode_utilise == "RAPIDE":
        return """
          <h2>Test de sensibilité (tornade)</h2>
          <p class="info">
            ⚡ <b>Mode RAPIDE</b> — tornade désactivée. Relance en mode
            COMPLET pour obtenir les scénarios de robustesse.
          </p>
        """
    rows = []
    for (m, c), info in stabilite.items():
        stable = info["stable"]
        badge = ('<span class="ok">STABLE</span>' if stable
                 else '<span class="err">FRAGILE</span>')
        flips = (", ".join(info["flips"]) if info["flips"] else "—")
        rows.append(f"""
          <tr>
            <td>{m} {c.upper()}</td>
            <td>{badge}</td>
            <td>{len(info['scenarios'])}</td>
            <td style="font-size:11px;">{flips}</td>
          </tr>
        """)
    return f"""
      <h2>Test de sensibilité (tornade)</h2>
      <p class="info">3 scénarios : moyenne ±15 %, retrait extrêmes.</p>
      <table>
        <tr><th>Ligne</th><th>Stabilité</th><th>Scénarios</th><th>Flips</th></tr>
        {''.join(rows)}
      </table>
    """


@app.post("/analyser", response_class=HTMLResponse)
async def analyser(request: Request):
    print("=== DEBUG /analyser : reception ===", flush=True)
    try:
        form = await request.form()
    except Exception as e:
        print(f"=== DEBUG : erreur form = {e}", flush=True)
        return _page("Erreur interne", f"""
          <h1 class="err">Erreur de réception</h1>
          <div class="card"><pre>{type(e).__name__} : {e}</pre></div>
        """)

    print(f"=== DEBUG : cles = {list(form.keys())}", flush=True)
    print(f"=== DEBUG : mode = {form.get('mode', 'ABSENT')}", flush=True)

    try:
        hist_home = parser_historique(form["home_history"], "H", form["home_team"], 15)
        print(f"=== DEBUG : hist_home OK ({len(hist_home)})", flush=True)
        hist_away = parser_historique(form["away_history"], "A", form["away_team"], 15)
        print(f"=== DEBUG : hist_away OK ({len(hist_away)})", flush=True)
        print("=== DEBUG : lancement analyser_match ===", flush=True)
        data = {
            "match": {
                "competition": form["competition"].strip(),
                "home_team": form["home_team"].strip(),
                "away_team": form["away_team"].strip(),
                "date": form["date"].strip(),
                "mode": form["mode"],
            },
            "home_team_history": hist_home,
            "away_team_history": hist_away,
            "market_lines": {
                "MC1": {"line": float(form["mc1_line"]),
                        "over": float(form["mc1_over"]),
                        "under": float(form["mc1_under"])},
                "MC2": {"line": float(form["mc2_line"]),
                        "over": float(form["mc2_over"]),
                        "under": float(form["mc2_under"])},
                "GLOBAL": {"line": float(form["glo_line"]),
                           "over": float(form["glo_over"]),
                           "under": float(form["glo_under"])},
            },
            "odds_1x2": {
                "home": float(form["odd_home"]),
                "draw": float(form["odd_draw"]),
                "away": float(form["odd_away"]),
            },
        }
        rapport = analyser_match(data)
        print("=== DEBUG : analyser_match OK ===", flush=True)
    except ValidationError as e:
        print(f"=== DEBUG : ValidationError = {e}", flush=True)
        return _page("Erreur de saisie", f"""
          <h1 class="err">Erreur de saisie</h1>
          <div class="card"><pre>{e}</pre></div>
          <p style="margin-top:20px;"><a href="/nouveau">← Corriger</a></p>
        """)
    except Exception as e:
        import traceback
        print("=== DEBUG : EXCEPTION ===", flush=True)
        print(traceback.format_exc(), flush=True)
        return _page("Erreur interne", f"""
          <h1 class="err">Erreur interne</h1>
          <div class="card"><pre>{type(e).__name__} : {e}</pre></div>
          <p style="margin-top:20px;"><a href="/nouveau">← Retour</a></p>
        """)

    verdict = rapport["verdict"]
    ok = bool(rapport["selections_acceptees"])
    classe = "verdict-oui" if ok else "verdict-non"
    match_label = (
        f"{rapport['match']['home_team']} vs {rapport['match']['away_team']} "
        f"({rapport['match']['date']})"
    )
    bloc_marches = "".join(
        _rendre_marche(nom, info, match_label)
        for nom, info in rapport["marches"].items()
    )
    stats = rapport["stats"]
    s_mc1 = stats["MC1"]; s_mc2 = stats["MC2"]; s_glo = stats["GLOBAL"]
    effet_score_html = _rendre_effet_score(rapport.get("score_effect"))
    dependance_html = _rendre_dependance_inter(rapport.get("dependance_inter"))
    zi_html = _rendre_zero_inflation(rapport.get("zero_inflation"))
    mode_utilise = rapport.get("mode_utilise", "COMPLET")
    tornade_html = _rendre_tornade(rapport.get("stabilite"), mode_utilise)
    ligue_style_html = _rendre_ligue_style(
        rapport.get("ligue"),
        rapport.get("style_home", {}),
        rapport.get("style_away", {}),
        rapport["match"]["home_team"],
        rapport["match"]["away_team"],
    )
    integrite_html = _rendre_integrite_calibration(
        rapport.get("integrite"),
        rapport.get("calibration_active", False),
        len(rapport.get("marches", {})),
    )
    print("=== DEBUG : rendu HTML OK ===", flush=True)

    return _page("Rapport", f"""
      <h1>Rapport</h1>
      <p class="info">
        {rapport['match']['home_team']} vs {rapport['match']['away_team']}
        — {rapport['match']['date']} — {rapport['match']['competition']}
        — <b>Mode {mode_utilise}</b>
      </p>
      <div class="verdict {classe}">{verdict}</div>

      {ligue_style_html}

      <h2>Statistiques du modèle</h2>
      <table>
        <tr><th>Marché</th><th>Moyenne</th><th>Médiane</th>
            <th>Q10</th><th>Q90</th><th>P(zéro)</th></tr>
        <tr><td>MC1</td><td>{s_mc1['mean']:.2f}</td><td>{s_mc1['median']}</td>
            <td>{s_mc1['q10']}</td><td>{s_mc1['q90']}</td>
            <td>{stats['p_zero_MC1']*100:.1f}%</td></tr>
        <tr><td>MC2</td><td>{s_mc2['mean']:.2f}</td><td>{s_mc2['median']}</td>
            <td>{s_mc2['q10']}</td><td>{s_mc2['q90']}</td>
            <td>{stats['p_zero_MC2']*100:.1f}%</td></tr>
        <tr><td>GLOBAL</td><td>{s_glo['mean']:.2f}</td><td>{s_glo['median']}</td>
            <td>{s_glo['q10']}</td><td>{s_glo['q90']}</td>
            <td>{stats['p_zero_GLOBAL']*100:.1f}%</td></tr>
      </table>

      {effet_score_html}

      {dependance_html}

      {zi_html}

      {tornade_html}

      {integrite_html}

      <h2>Détail par marché</h2>
      {bloc_marches}

      <h2>Contexte</h2>
      <p class="info">
        Dépendance A1/A2 domicile :
        τ = {rapport['copule_home']['tau']:.3f}, θ = {rapport['copule_home']['theta']:.3f}
        ({'indépendance retenue' if rapport['copule_home']['independance_ok'] else 'dépendance détectée'})<br>
        Dépendance A1/A2 extérieur :
        τ = {rapport['copule_away']['tau']:.3f}, θ = {rapport['copule_away']['theta']:.3f}
        ({'indépendance retenue' if rapport['copule_away']['independance_ok'] else 'dépendance détectée'})<br>
        PageRank homogène : {'oui' if rapport['pagerank']['homogene'] else 'non'}
      </p>

      <p style="margin-top:24px;"><a href="/nouveau">← Nouvelle analyse</a></p>
    """, scripts='<script src="/static/suivi.js"></script>')


@app.get("/health")
def health():
    return {"status": "ok", "service": "corners-pro", "version": "1.3.1"}
