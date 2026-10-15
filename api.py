"""
Serveur web du bot corners-pro.
Version 1 : endpoint minimal pour verifier que le deploiement fonctionne.
Version finale : formulaire + analyse (ajoute apres validation du deploiement).
"""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse


app = FastAPI(title="corners-pro", version="0.1.0")


@app.get("/", response_class=HTMLResponse)
def racine():
    return """
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>corners-pro</title>
      <style>
        body { font-family: -apple-system, sans-serif; background:#0d1117; color:#c9d1d9;
               margin:0; padding:40px 20px; text-align:center; }
        h1 { color:#58a6ff; }
        .ok { display:inline-block; background:#238636; color:white; padding:8px 16px;
              border-radius:8px; font-weight:bold; }
        .info { margin-top:20px; color:#8b949e; font-size:14px; }
      </style>
    </head>
    <body>
      <h1>corners-pro</h1>
      <div class="ok">🟢 Bot en ligne</div>
      <div class="info">
        Version 0.1.0 — serveur de base fonctionnel.<br>
        Le formulaire arrive a l'etape suivante.
      </div>
    </body>
    </html>
    """


@app.get("/health")
def health():
    """Endpoint technique pour verifier que le serveur repond."""
    return {"status": "ok", "service": "corners-pro"}
