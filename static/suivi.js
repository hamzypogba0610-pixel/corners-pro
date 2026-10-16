// Suivi des predictions (localStorage navigateur).
// Structure : liste d'objets {id, date, match, marche, cote, p_estime,
//   ev_prevu, verdict, resultat_reel (null au depart)}

const CLE_SUIVI = "corners_pro_suivi_v1";

function lireSuivi() {
  try {
    const raw = localStorage.getItem(CLE_SUIVI);
    if (!raw) return [];
    return JSON.parse(raw) || [];
  } catch (e) {
    console.error("Erreur lecture suivi :", e);
    return [];
  }
}

function ecrireSuivi(preds) {
  try {
    localStorage.setItem(CLE_SUIVI, JSON.stringify(preds));
    return true;
  } catch (e) {
    alert("Erreur sauvegarde suivi : " + e.message);
    return false;
  }
}

function enregistrerPrediction(obj) {
  const preds = lireSuivi();
  const id = Date.now() + "_" + Math.random().toString(36).slice(2, 8);
  obj.id = id;
  obj.date_enregistrement = new Date().toISOString();
  obj.resultat_reel = null;  // "gagne" / "perdu" / "push"
  preds.unshift(obj);
  if (ecrireSuivi(preds)) {
    alert("✅ Prédiction enregistrée.");
    return true;
  }
  return false;
}

function mettreAJourResultat(id, resultat) {
  const preds = lireSuivi();
  const idx = preds.findIndex(p => p.id === id);
  if (idx < 0) return false;
  preds[idx].resultat_reel = resultat;
  return ecrireSuivi(preds);
}

function supprimerPrediction(id) {
  if (!confirm("Supprimer cette prédiction ?")) return false;
  const preds = lireSuivi().filter(p => p.id !== id);
  return ecrireSuivi(preds);
}

function exporterSuivi() {
  const data = JSON.stringify(lireSuivi(), null, 2);
  const blob = new Blob([data], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "corners-pro-suivi-" + new Date().toISOString().slice(0,10) + ".json";
  a.click();
  URL.revokeObjectURL(url);
}

function importerSuivi(input) {
  const f = input.files && input.files[0];
  if (!f) return;
  const reader = new FileReader();
  reader.onload = e => {
    try {
      const incoming = JSON.parse(e.target.result);
      if (!Array.isArray(incoming)) throw new Error("Format invalide");
      const current = lireSuivi();
      const existingIds = new Set(current.map(p => p.id));
      let n = 0;
      for (const p of incoming) {
        if (p && p.id && !existingIds.has(p.id)) {
          current.unshift(p);
          n++;
        }
      }
      ecrireSuivi(current);
      alert("✅ " + n + " prédiction(s) importée(s).");
      location.reload();
    } catch (err) {
      alert("Erreur import : " + err.message);
    }
  };
  reader.readAsText(f);
}

// Marquer le resultat depuis la liste
function setResultat(id, res) {
  if (mettreAJourResultat(id, res)) {
    location.reload();
  }
}

// Enregistrer une prediction depuis le rapport
function enregistrerDepuisRapport(btn) {
  const obj = {
    match: btn.dataset.match,
    date: btn.dataset.date,
    marche: btn.dataset.marche,
    ligne: btn.dataset.ligne,
    pari: btn.dataset.pari,
    cote: parseFloat(btn.dataset.cote),
    p_estime: parseFloat(btn.dataset.p),
    ev_prevu: parseFloat(btn.dataset.ev),
  };
  if (enregistrerPrediction(obj)) {
    btn.textContent = "✓ Enregistré";
    btn.disabled = true;
    btn.style.opacity = "0.5";
  }
}

// Calcule les stats sur les predictions terminees
function calculerStatsSuivi() {
  const preds = lireSuivi();
  const terminees = preds.filter(p =>
    p.resultat_reel === "gagne" || p.resultat_reel === "perdu"
  );
  if (terminees.length === 0) {
    return { n: 0 };
  }
  let gains = 0, pertes = 0, ev_prevu_total = 0, ev_reel_total = 0;
  let brier = 0;
  for (const p of terminees) {
    const y = p.resultat_reel === "gagne" ? 1 : 0;
    brier += Math.pow(p.p_estime - y, 2);
    ev_prevu_total += p.ev_prevu;
    if (y === 1) {
      gains += (p.cote - 1);
      ev_reel_total += (p.cote - 1);
    } else {
      pertes += 1;
      ev_reel_total -= 1;
    }
  }
  const mise_totale = terminees.length;
  return {
    n: terminees.length,
    brier: brier / terminees.length,
    ev_prevu_moyen: ev_prevu_total / terminees.length,
    ev_reel_moyen: ev_reel_total / mise_totale,
    pnl: ev_reel_total,
    taux_reussite: gains / (gains + pertes) * 100,
    roi: (ev_reel_total / mise_totale) * 100,
  };
    }
