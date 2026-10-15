"""
Serveur web du bot corners-pro.
Version 5 : sauvegarde locale des equipes (localStorage).
"""
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os

from data.schema import ValidationError
from core.analysis import parser_historique, analyser_match


app = FastAPI(title="corners-pro", version="0.5.0")

# Servir les fichiers statiques (JS, CSS)
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
        <div class="info">Version 0.5.0 — sauvegarde des équipes</div>
      </div>
      <p style="margin-top:24px;"><a href="/nouveau">→ Lancer une nouvelle analyse</a></p>
      <p style="margin-top:8px;"><a href="/equipes">→ Mes équipes sauvegardées</a></p>
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
          <option value="COMPLET">COMPLET</option>
          <option value="RAPIDE">RAPIDE</option>
        </select>

        <h2>2. Cotes 1X2 du match</h2>
        <label>Cote Domicile</label>
        <input name="odd_home" type="number" step="0.01" min="1.01" required placeholder="2.10">
        <label>Cote Nul</label>
        <input name="odd_draw" type="number" step="0.01" min="1.01" required placeholder="3.40">
        <label>Cote Extérieur</label>
        <input name="odd_away" type="number" step="0.01" min="1.01" required placeholder="3.20">

        <h2>3. Historique équipe domicile (15 matchs à domicile)</h2>
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

        <h2>4. Historique équipe extérieur (15 matchs à l'extérieur)</h2>
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
        <div class="info">Ces équipes sont stockées dans <b>ce navigateur</b>.
        Si tu changes de téléphone, exporte d'abord.</div>
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


def _fmt_pct(x):
    return f"{x*100:.2f}%"


def _fmt_ev(x):
    return f"{x:+.4f}"


def _rendre_ligne_pari(nom, cote_nom, c):
    label = cote_nom.upper()
    gate = c["gate"]
    couleur = "ok" if gate["accepte"] else "err"
    statut = "✅ RETENU" if gate["accepte"] else "❌ REFUSÉ"
    raisons_html = ""
    if not gate["accepte"] and gate["raisons_refus"]:
        items = "".join(f"<li>{r}</li>" for r in gate["raisons_refus"])
        raisons_html = f'<div class="raisons">Pourquoi ce refus ?<ul>{items}</ul></div>'
    return f"""
      <tr>
        <td><b>{label}</b></td>
        <td>{_fmt_pct(c['p'])}</td>
        <td>{_fmt_pct(c['p_impl_prop'])}</td>
        <td>{c['cote']:.2f}</td>
        <td>{_fmt_ev(c['ev_info']['ev_mediane'])} / {_fmt_ev(c['ev_info']['ev_p20'])}</td>
        <td>{c['kelly']*100:.2f}%</td>
        <td class="{couleur}">{statut}</td>
      </tr>
      <tr><td colspan="7" style="padding:0; border:none;">{raisons_html}</td></tr>
    """


def _rendre_marche(nom, info):
    rows = (
        _rendre_ligne_pari(nom, "over", info["cotes"]["over"])
        + _rendre_ligne_pari(nom, "under", info["cotes"]["under"])
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


@app.post("/analyser", response_class=HTMLResponse)
async def analyser(request: Request):
    form = await request.form()
    try:
        hist_home = parser_historique(form["home_history"], "H", form["home_team"], 15)
        hist_away = parser_historique(form["away_history"], "A", form["away_team"], 15)
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
    except ValidationError as e:
        return _page("Erreur de saisie", f"""
          <h1 class="err">Erreur de saisie</h1>
          <div class="card"><pre>{e}</pre></div>
          <p style="margin-top:20px;"><a href="/nouveau">← Corriger</a></p>
        """)
    except Exception as e:
        return _page("Erreur interne", f"""
          <h1 class="err">Erreur interne</h1>
          <div class="card"><pre>{type(e).__name__} : {e}</pre></div>
          <p style="margin-top:20px;"><a href="/nouveau">← Retour</a></p>
        """)

    verdict = rapport["verdict"]
    ok = bool(rapport["selections_acceptees"])
    classe = "verdict-oui" if ok else "verdict-non"
    bloc_marches = "".join(
        _rendre_marche(nom, info) for nom, info in rapport["marches"].items()
    )
    stats = rapport["stats"]
    s_mc1 = stats["MC1"]; s_mc2 = stats["MC2"]; s_glo = stats["GLOBAL"]

    return _page("Rapport", f"""
      <h1>Rapport</h1>
      <p class="info">
        {rapport['match']['home_team']} vs {rapport['match']['away_team']}
        — {rapport['match']['date']} — {rapport['match']['competition']}
      </p>
      <div class="verdict {classe}">{verdict}</div>

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
    """)


@app.get("/health")
def health():
    return {"status": "ok", "service": "corners-pro", "version": "0.5.0"}
