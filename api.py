"""
Serveur web du bot corners-pro.
Version 2 : formulaire de saisie (avec textareas CSV).
"""
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse


app = FastAPI(title="corners-pro", version="0.2.0")


def _page(titre, corps):
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
    a {{ color:#58a6ff; }}
    label {{ display:block; margin:12px 0 4px; color:#8b949e; font-size:13px; }}
    input, select, textarea {{
      width:100%; padding:10px; border-radius:8px;
      border:1px solid #30363d; background:#161b22; color:#c9d1d9;
      font-family:inherit; font-size:15px;
    }}
    textarea {{ min-height:220px; font-family:monospace; font-size:12px; }}
    button {{
      margin-top:24px; width:100%; padding:14px;
      background:#238636; color:white; border:none;
      border-radius:8px; font-size:16px; font-weight:bold;
    }}
    button:active {{ background:#2ea043; }}
    .card {{ background:#161b22; border:1px solid #30363d; border-radius:10px;
             padding:14px; margin-top:12px; }}
    .ok {{ color:#3fb950; font-weight:bold; }}
    .err {{ color:#f85149; }}
    .info {{ color:#8b949e; font-size:13px; margin:6px 0; }}
    code {{ background:#21262d; padding:2px 6px; border-radius:4px; font-size:11px; }}
  </style>
</head>
<body>
{corps}
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def racine():
    return _page("corners-pro", """
      <h1>corners-pro</h1>
      <div class="card">
        <span class="ok">🟢 Bot en ligne</span>
        <div class="info">Version 0.2.0 — formulaire disponible</div>
      </div>
      <p style="margin-top:24px;">
        <a href="/nouveau">→ Lancer une nouvelle analyse</a>
      </p>
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
        <input name="home_team" required placeholder="Paris">
        <label>Équipe extérieur</label>
        <input name="away_team" required placeholder="Lyon">
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

        <h2>3. Historique équipe domicile</h2>
        <div class="info">
          <b>15 matchs à domicile</b>, un par ligne. Format :<br>
          <code>date,adversaire,c1h,c2h,rf,rmt,rouge,xg,xga,xgot,xgota</code><br>
          rf/rmt = V/N/D — rouge = 0/1/2 — xg = nombres décimaux
        </div>
        <textarea name="home_history" required placeholder="2026-01-15,Marseille,3,4,V,N,0,1.5,0.9,1.2,0.7
2026-01-22,Lyon,2,3,D,N,1,1.1,1.4,0.8,1.0"></textarea>

        <h2>4. Historique équipe extérieur</h2>
        <div class="info">
          <b>15 matchs à l'extérieur</b>, même format.
        </div>
        <textarea name="away_history" required placeholder="2026-01-15,Nice,2,2,N,V,0,1.0,1.2,0.8,0.9"></textarea>

        <h2>5. Cotes des marchés</h2>
        <label>MC1 — ligne (ex: 12.5)</label>
        <input name="mc1_line" type="number" step="0.5" required placeholder="12.5">
        <label>MC1 — cote Over</label>
        <input name="mc1_over" type="number" step="0.01" min="1.01" required placeholder="1.90">
        <label>MC1 — cote Under</label>
        <input name="mc1_under" type="number" step="0.01" min="1.01" required placeholder="1.90">

        <label>MC2 — ligne</label>
        <input name="mc2_line" type="number" step="0.5" required placeholder="12.5">
        <label>MC2 — cote Over</label>
        <input name="mc2_over" type="number" step="0.01" min="1.01" required placeholder="1.85">
        <label>MC2 — cote Under</label>
        <input name="mc2_under" type="number" step="0.01" min="1.01" required placeholder="1.95">

        <label>GLOBAL — ligne</label>
        <input name="glo_line" type="number" step="0.5" required placeholder="25.5">
        <label>GLOBAL — cote Over</label>
        <input name="glo_over" type="number" step="0.01" min="1.01" required placeholder="1.88">
        <label>GLOBAL — cote Under</label>
        <input name="glo_under" type="number" step="0.01" min="1.01" required placeholder="1.92">

        <button type="submit">Analyser le match</button>
      </form>
    """)


@app.post("/analyser", response_class=HTMLResponse)
async def analyser(request: Request):
    form = await request.form()
    lignes = "".join(
        f"<tr><td style='padding:6px; border-bottom:1px solid #21262d;'>{k}</td>"
        f"<td style='padding:6px; border-bottom:1px solid #21262d;'>"
        f"<code>{(v[:100] + '...') if isinstance(v, str) and len(v) > 100 else v}</code>"
        f"</td></tr>"
        for k, v in form.items()
    )
    return _page("Reçu", f"""
      <h1>Données reçues</h1>
      <p class="ok">✓ Le formulaire a bien été soumis.</p>
      <p class="info">On câblera l'analyse à l'étape suivante.</p>
      <table style="width:100%; margin-top:16px; font-size:12px; border-collapse:collapse;">
        {lignes}
      </table>
      <p style="margin-top:20px;"><a href="/nouveau">← Retour au formulaire</a></p>
    """)


@app.get("/health")
def health():
    return {"status": "ok", "service": "corners-pro"}
