// Gestion locale des equipes (localStorage du navigateur).
// Structure : { "Paris": {venue: "H", lignes: "..."}, ... }

const CLE_STOCKAGE = "corners_pro_equipes_v1";

function lireEquipes() {
  try {
    const raw = localStorage.getItem(CLE_STOCKAGE);
    if (!raw) return {};
    return JSON.parse(raw) || {};
  } catch (e) {
    console.error("Erreur lecture equipes :", e);
    return {};
  }
}

function ecrireEquipes(equipes) {
  try {
    localStorage.setItem(CLE_STOCKAGE, JSON.stringify(equipes));
    return true;
  } catch (e) {
    alert("Erreur sauvegarde : " + e.message);
    return false;
  }
}

function listerNoms() {
  return Object.keys(lireEquipes()).sort();
}

function sauvegarderEquipe(nom, venue, lignes) {
  nom = (nom || "").trim();
  if (!nom) { alert("Le nom de l'equipe est vide."); return false; }
  if (!lignes || !lignes.trim()) { alert("Aucune donnee a sauvegarder."); return false; }
  const equipes = lireEquipes();
  equipes[nom] = { venue: venue, lignes: lignes.trim(), date_sauvegarde: new Date().toISOString() };
  if (ecrireEquipes(equipes)) {
    alert("✅ Equipe \"" + nom + "\" sauvegardee.");
    return true;
  }
  return false;
}

function chargerEquipe(nom) {
  const equipes = lireEquipes();
  return equipes[nom] || null;
}

function supprimerEquipe(nom) {
  if (!confirm("Supprimer \"" + nom + "\" ?")) return false;
  const equipes = lireEquipes();
  delete equipes[nom];
  return ecrireEquipes(equipes);
}

// --- Actions liees au formulaire /nouveau ---

function sauvegarderDepuisTextarea(idTextarea, idInputNom) {
  const ta = document.getElementById(idTextarea);
  const inp = document.getElementById(idInputNom);
  if (!ta || !inp) return;
  const venue = idTextarea.includes("home") ? "H" : "A";
  sauvegarderEquipe(inp.value, venue, ta.value);
}

function remplirDepuisListe(idTextarea, idSelect) {
  const ta = document.getElementById(idTextarea);
  const sel = document.getElementById(idSelect);
  if (!ta || !sel) return;
  const nom = sel.value;
  if (!nom) return;
  const eq = chargerEquipe(nom);
  if (!eq) return;
  ta.value = eq.lignes;
}

function peuplerSelect(idSelect, venue) {
  const sel = document.getElementById(idSelect);
  if (!sel) return;
  const equipes = lireEquipes();
  const noms = Object.keys(equipes)
    .filter(n => !venue || equipes[n].venue === venue)
    .sort();
  sel.innerHTML = '<option value="">— Choisir une equipe —</option>';
  noms.forEach(n => {
    const o = document.createElement("option");
    o.value = n;
    o.textContent = n + " (" + equipes[n].venue + ")";
    sel.appendChild(o);
  });
}

// --- Export / Import global ---

function exporterTout() {
  const data = localStorage.getItem(CLE_STOCKAGE) || "{}";
  const blob = new Blob([data], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "corners-pro-equipes-" + new Date().toISOString().slice(0,10) + ".json";
  a.click();
  URL.revokeObjectURL(url);
}

function importerFichier(input) {
  const f = input.files && input.files[0];
  if (!f) return;
  const reader = new FileReader();
  reader.onload = e => {
    try {
      const incoming = JSON.parse(e.target.result);
      const current = lireEquipes();
      let n = 0;
      for (const [nom, val] of Object.entries(incoming)) {
        if (val && val.lignes) { current[nom] = val; n++; }
      }
      ecrireEquipes(current);
      alert("✅ " + n + " equipe(s) importee(s).");
      location.reload();
    } catch (err) {
      alert("Erreur import : " + err.message);
    }
  };
  reader.readAsText(f);
      }
