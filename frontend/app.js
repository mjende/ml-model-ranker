const $ = (sel) => document.querySelector(sel);

const CRITERIA = [
  { key: "popularity", label: "Popularność" },
  { key: "architecture", label: "Architektura" },
  { key: "size", label: "Rozmiar (mniejszy = lepszy)" },
  { key: "accuracy", label: "Precyzja / Jakość" },
  { key: "speed", label: "Szybkość inferencji" },
  { key: "documentation", label: "Dokumentacja / Wsparcie" },
];

let defaultWeights = {};
let lastRankedRows = null;

async function loadDefaults() {
  const r = await fetch("/api/weights");
  defaultWeights = await r.json();
  renderWeights(defaultWeights);
}

function renderWeights(values) {
  const root = $("#weights");
  root.innerHTML = "";
  for (const c of CRITERIA) {
    const v = (values[c.key] ?? 0).toFixed(2);
    const id = `w-${c.key}`;
    const item = document.createElement("div");
    item.className = "weight-item";
    item.innerHTML = `
      <label for="${id}">${c.label} <span class="val" id="${id}-val">${v}</span></label>
      <input id="${id}" data-key="${c.key}" type="range" min="0" max="1" step="0.05" value="${v}">
    `;
    root.appendChild(item);
    item.querySelector("input").addEventListener("input", (e) => {
      $(`#${id}-val`).textContent = parseFloat(e.target.value).toFixed(2);
    });
  }
}

function readWeights() {
  const out = {};
  document.querySelectorAll(".weight-item input[type=range]").forEach((el) => {
    out[el.dataset.key] = parseFloat(el.value);
  });
  return out;
}

function setStatus(msg, kind = "") {
  const el = $("#status");
  el.textContent = msg;
  el.className = "status " + kind;
}

function escapeHtml(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function renderTable(rows) {
  if (!rows || !rows.length) {
    $("#results").innerHTML = '<p class="empty">Brak wyników.</p>';
    return;
  }
  const shown = ["rank", "model", "score", "justification"];
  const normCols = Object.keys(rows[0]).filter((k) => k.endsWith("_norm"));
  const headers = [...shown, ...normCols];
  const head = headers.map((h) => `<th>${escapeHtml(h)}</th>`).join("");
  const maxScore = Math.max(...rows.map((r) => r.score || 0), 1);
  const body = rows
    .map((r) => {
      const cells = headers.map((h) => {
        let v = r[h];
        if (v === null || v === undefined || (typeof v === "number" && Number.isNaN(v))) v = "";
        if (h === "rank") return `<td class="rank">${escapeHtml(v)}</td>`;
        if (h === "score") {
          const pct = ((r.score || 0) / maxScore) * 100;
          return `<td class="score-cell">${escapeHtml(v)}<div class="score-bar"><div style="width:${pct}%"></div></div></td>`;
        }
        if (typeof v === "number") v = Number.isInteger(v) ? v : v.toFixed(3);
        return `<td>${escapeHtml(v)}</td>`;
      });
      return `<tr>${cells.join("")}</tr>`;
    })
    .join("");
  $("#results").innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

async function runRanking() {
  const fileEl = $("#file");
  if (!fileEl.files.length) {
    setStatus("Wybierz plik CSV.", "error");
    return;
  }
  setStatus("Obliczanie...");
  $("#download").disabled = true;

  const fd = new FormData();
  fd.append("file", fileEl.files[0]);
  fd.append("weights", JSON.stringify(readWeights()));
  fd.append("enrich", $("#enrich").checked ? "true" : "false");

  try {
    const r = await fetch("/api/rank", { method: "POST", body: fd });
    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: r.statusText }));
      throw new Error(err.detail || "Błąd serwera");
    }
    const data = await r.json();
    lastRankedRows = data.rows;
    renderTable(data.rows);
    setStatus(`Posortowano ${data.rows.length} modeli.`, "ok");
    $("#download").disabled = false;
  } catch (e) {
    setStatus("Błąd: " + e.message, "error");
  }
}

async function downloadCsv() {
  const fileEl = $("#file");
  if (!fileEl.files.length) {
    setStatus("Najpierw wgraj plik CSV.", "error");
    return;
  }
  const fd = new FormData();
  fd.append("file", fileEl.files[0]);
  fd.append("weights", JSON.stringify(readWeights()));
  fd.append("enrich", $("#enrich").checked ? "true" : "false");
  const r = await fetch("/api/rank/csv", { method: "POST", body: fd });
  if (!r.ok) {
    setStatus("Błąd pobierania CSV", "error");
    return;
  }
  const blob = await r.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "models_ranked.csv";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

document.addEventListener("DOMContentLoaded", () => {
  loadDefaults();
  $("#run").addEventListener("click", runRanking);
  $("#download").addEventListener("click", downloadCsv);
  $("#reset-weights").addEventListener("click", () => renderWeights(defaultWeights));
});
