// ML Model Ranker — frontend
'use strict';

const LS_WEIGHTS = 'mlmr.weights.v2';
const LS_PRESET = 'mlmr.preset.v2';
const LS_ENRICH = 'mlmr.enrich.v2';

let DEFAULT_WEIGHTS = {};
let PRESETS = {};
let lastRows = [];
let lastColumns = [];
let sortState = { col: 'rank', dir: 'asc' };

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

async function fetchJSON(url, init) {
  const r = await fetch(url, init);
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try { const j = await r.json(); if (j.detail) msg = j.detail; else if (j.error) msg = j.error; } catch (_) {}
    throw new Error(msg);
  }
  return r.json();
}

// ----- weights UI -----
function renderWeights(weights) {
  const grid = $('#weights');
  grid.innerHTML = '';
  const labels = {
    popularity: 'popularność',
    architecture: 'architektura',
    size: 'rozmiar',
    accuracy: 'jakość',
    speed: 'szybkość',
    documentation: 'dokumentacja',
    recency: 'świeżość',
    license: 'licencja',
  };
  for (const key of Object.keys(weights)) {
    const row = document.createElement('div');
    row.className = 'weight';
    const v = weights[key];
    row.innerHTML = `
      <label for="w-${key}">${labels[key] || key}</label>
      <input type="range" id="w-${key}" data-key="${key}" min="0" max="1" step="0.01" value="${v}" />
      <span class="val" id="v-${key}">${Number(v).toFixed(2)}</span>
    `;
    grid.appendChild(row);
  }
  grid.addEventListener('input', (ev) => {
    const inp = ev.target;
    if (inp.tagName !== 'INPUT') return;
    const key = inp.dataset.key;
    $(`#v-${key}`).textContent = Number(inp.value).toFixed(2);
    $('#preset').value = 'custom';
    persistWeights();
  });
}

function readWeights() {
  const w = {};
  for (const inp of $$('#weights input[type="range"]')) {
    w[inp.dataset.key] = parseFloat(inp.value);
  }
  return w;
}

function setWeights(w) {
  for (const key of Object.keys(w)) {
    const inp = document.getElementById('w-' + key);
    if (inp) {
      inp.value = w[key];
      const val = document.getElementById('v-' + key);
      if (val) val.textContent = Number(w[key]).toFixed(2);
    }
  }
}

function persistWeights() {
  try { localStorage.setItem(LS_WEIGHTS, JSON.stringify(readWeights())); } catch (_) {}
  try { localStorage.setItem(LS_PRESET, $('#preset').value); } catch (_) {}
}

// ----- presets -----
function fillPresetSelect() {
  const sel = $('#preset');
  // keep "custom" entry
  for (const name of Object.keys(PRESETS)) {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = presetLabel(name);
    sel.appendChild(opt);
  }
  sel.addEventListener('change', () => {
    const name = sel.value;
    if (name === 'custom') { persistWeights(); return; }
    if (PRESETS[name]) {
      setWeights(PRESETS[name]);
      persistWeights();
    }
  });
}
function presetLabel(name) {
  const m = {
    balanced: 'Zbalansowany',
    qa_focus: 'QA focus (jakość + popularność)',
    performance: 'Wydajność (rozmiar + szybkość)',
    research: 'Research (architektura + świeżość)',
    production: 'Produkcja (licencja + popularność)',
  };
  return m[name] || name;
}

// ----- main run -----
async function runRanking() {
  const fileInput = $('#file');
  if (!fileInput.files.length) {
    setStatus('Wybierz najpierw plik CSV / XLSX.', 'error');
    return;
  }
  const status = $('#status');
  const enrich = $('#enrich').checked;
  const weights = readWeights();
  try { localStorage.setItem(LS_ENRICH, enrich ? '1' : '0'); } catch (_) {}

  const fd = new FormData();
  fd.append('file', fileInput.files[0]);
  fd.append('enrich', enrich ? 'true' : 'false');
  fd.append('weights', JSON.stringify(weights));

  $('#run').disabled = true;
  $('#download-csv').disabled = true;
  $('#download-json').disabled = true;
  setStatus('');
  $('#progress').hidden = !enrich;

  try {
    const t0 = performance.now();
    const data = await fetchJSON('/api/rank', { method: 'POST', body: fd });
    const dt = ((performance.now() - t0) / 1000).toFixed(1);
    lastRows = data.rows || [];
    lastColumns = data.columns || [];
    sortState = { col: 'rank', dir: 'asc' };
    populateModalityFilter(lastRows);
    renderResults();
    $('#results-panel').hidden = false;
    $('#empty-state').hidden = true;
    $('#results-count').textContent = `${lastRows.length} modeli · ${dt}s`;
    $('#download-csv').disabled = false;
    $('#download-json').disabled = false;
    setStatus('✓ Gotowe', 'ok');
  } catch (e) {
    setStatus('Błąd: ' + e.message, 'error');
  } finally {
    $('#run').disabled = false;
    $('#progress').hidden = true;
  }
}

function setStatus(msg, kind) {
  const s = $('#status');
  s.textContent = msg || '';
  s.className = 'status' + (kind ? ' ' + kind : '');
}

// ----- results rendering -----
const PRIMARY_COLS = [
  'rank', 'model_name', 'score', 'coverage',
  'downloads', 'likes', 'parameters', 'architecture_detected',
  'license', 'last_modified', 'pipeline_tag', 'justification'
];
const COL_LABELS = {
  rank: '#',
  model_name: 'Model',
  score: 'Wynik',
  coverage: 'Pokrycie',
  downloads: 'Pobrania HF',
  likes: 'Likes',
  parameters: 'Parametry',
  architecture_detected: 'Architektura',
  license: 'Licencja',
  last_modified: 'Aktualizacja',
  pipeline_tag: 'Pipeline',
  justification: 'Uzasadnienie',
};

function visibleColumns() {
  const showNorm = $('#show-norm').checked;
  const cols = PRIMARY_COLS.filter(c => lastColumns.includes(c));
  if (showNorm) {
    for (const c of lastColumns) {
      if (c.endsWith('_norm') && !cols.includes(c)) cols.push(c);
    }
  }
  return cols;
}

function getFilteredRows() {
  const q = ($('#search').value || '').toLowerCase().trim();
  const modality = ($('#modality-filter').value || '').toLowerCase();
  return lastRows.filter(r => {
    if (q) {
      const hay = [
        r.model_name, r.architecture_detected, r.architecture,
        r.license, r.pipeline_tag, r.hf_id, r.model
      ].filter(Boolean).join(' ').toLowerCase();
      if (!hay.includes(q)) return false;
    }
    if (modality) {
      const mod = (r['input_|_output_modalities'] || '').toString().toLowerCase();
      if (!mod.includes(modality)) return false;
    }
    return true;
  });
}

function sortRows(rows) {
  const { col, dir } = sortState;
  const sign = dir === 'asc' ? 1 : -1;
  return rows.slice().sort((a, b) => {
    const va = a[col], vb = b[col];
    const na = typeof va === 'number' ? va : (va == null ? null : parseFloat(va));
    const nb = typeof vb === 'number' ? vb : (vb == null ? null : parseFloat(vb));
    const aNum = !Number.isNaN(na) && na !== null && va !== '' && (typeof va === 'number' || !isNaN(Number(va)));
    const bNum = !Number.isNaN(nb) && nb !== null && vb !== '' && (typeof vb === 'number' || !isNaN(Number(vb)));
    if (aNum && bNum) return (na - nb) * sign;
    const sa = (va == null ? '' : String(va)).toLowerCase();
    const sb = (vb == null ? '' : String(vb)).toLowerCase();
    if (sa < sb) return -1 * sign;
    if (sa > sb) return 1 * sign;
    return 0;
  });
}

function renderResults() {
  const container = $('#results');
  const cols = visibleColumns();
  const rows = sortRows(getFilteredRows());

  const thead = '<thead><tr>' + cols.map(c => {
    const cls = c === sortState.col ? 'sort-' + sortState.dir : '';
    return `<th data-col="${c}" class="${cls}">${COL_LABELS[c] || c}</th>`;
  }).join('') + '</tr></thead>';

  const tbody = '<tbody>' + rows.map((r, idx) => {
    const rankCls = r.rank === 1 ? 'top-1' : r.rank === 2 ? 'top-2' : r.rank === 3 ? 'top-3' : '';
    return `<tr class="${rankCls}" data-idx="${lastRows.indexOf(r)}">` +
      cols.map(c => `<td class="${c === 'rank' ? 'rank' : ''} ${c === 'justification' ? 'justify' : ''}">${formatCell(c, r[c], r)}</td>`).join('') +
      '</tr>';
  }).join('') + '</tbody>';

  container.innerHTML = '<table>' + thead + tbody + '</table>';

  // header click → sort
  for (const th of container.querySelectorAll('th')) {
    th.addEventListener('click', () => {
      const col = th.dataset.col;
      if (sortState.col === col) {
        sortState.dir = sortState.dir === 'asc' ? 'desc' : 'asc';
      } else {
        sortState.col = col;
        sortState.dir = (col === 'model_name' || col === 'license' || col === 'pipeline_tag') ? 'asc' : 'desc';
      }
      renderResults();
    });
  }

  // row click → modal
  for (const tr of container.querySelectorAll('tbody tr')) {
    tr.addEventListener('click', () => {
      const idx = parseInt(tr.dataset.idx, 10);
      if (!isNaN(idx)) openModal(lastRows[idx]);
    });
  }
}

function formatCell(col, val, row) {
  if (val === null || val === undefined || val === '') return '<span style="color:#cbd5e1">—</span>';
  if (col === 'score') {
    const v = Number(val);
    const pct = Math.max(0, Math.min(100, v));
    const color = scoreColor(pct);
    return `<div class="score-cell">
      <div class="score-bar"><div style="width:${pct}%;background:${color}"></div></div>
      <span class="score-value">${v.toFixed(1)}</span>
    </div>`;
  }
  if (col === 'coverage') {
    const v = Number(val);
    const cls = v >= 0.7 ? 'cov-high' : v >= 0.4 ? 'cov-mid' : 'cov-low';
    return `<span class="cov-badge ${cls}">${Math.round(v * 100)}%</span>`;
  }
  if (col === 'downloads' || col === 'likes' || col === 'parameters') {
    return formatNumber(Number(val));
  }
  if (col === 'last_modified') {
    return String(val).slice(0, 10);
  }
  if (col === 'rank') return String(val);
  if (col === 'justification') return escapeHtml(String(val));
  if (typeof val === 'number') {
    return Number.isFinite(val) ? val.toFixed(3) : String(val);
  }
  const s = String(val);
  return escapeHtml(s.length > 90 ? s.slice(0, 87) + '…' : s);
}

function formatNumber(n) {
  if (!Number.isFinite(n)) return '—';
  if (n >= 1e9) return (n / 1e9).toFixed(1) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'k';
  return String(Math.round(n));
}

function scoreColor(v) {
  // 0=red, 50=yellow, 100=green
  const h = Math.round((v / 100) * 120); // 0..120 hue
  return `hsl(${h}, 65%, 48%)`;
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

// ----- modal -----
function openModal(row) {
  const body = $('#modal-body');
  const hfId = row.hf_id || (row.model && /\//.test(row.model) ? row.model : null);
  const links = [];
  if (hfId) links.push(`<a href="https://huggingface.co/${encodeURIComponent(hfId)}" target="_blank" rel="noopener">Hugging Face ↗</a>`);
  if (row.github) {
    const ghPath = String(row.github).replace(/^https?:\/\/github\.com\//, '');
    links.push(`<a href="https://github.com/${ghPath}" target="_blank" rel="noopener">GitHub ↗</a>`);
  }

  const skip = new Set(['justification']);
  const entries = Object.entries(row)
    .filter(([k, v]) => !skip.has(k) && v !== null && v !== undefined && v !== '')
    .map(([k, v]) => `<dt>${escapeHtml(COL_LABELS[k] || k)}</dt><dd>${escapeHtml(String(v))}</dd>`)
    .join('');

  body.innerHTML = `
    <h3>#${row.rank} · ${escapeHtml(String(row.model_name || row.model || '—'))}</h3>
    <div class="subtle">Wynik: <strong>${Number(row.score).toFixed(1)}</strong> · pokrycie ${Math.round((row.coverage || 0) * 100)}%</div>
    <div class="links">${links.join('') || '<span class="subtle">brak linków</span>'}</div>
    <div class="justification-box">${escapeHtml(String(row.justification || ''))}</div>
    <dl>${entries}</dl>
  `;
  $('#modal').hidden = false;
}
function closeModal() { $('#modal').hidden = true; }

// ----- modality filter (from data) -----
function populateModalityFilter(rows) {
  const sel = $('#modality-filter');
  sel.innerHTML = '<option value="">Wszystkie modality</option>';
  const set = new Set();
  for (const r of rows) {
    const v = r['input_|_output_modalities'];
    if (v) set.add(String(v).trim());
  }
  const arr = Array.from(set).sort();
  for (const v of arr) {
    const o = document.createElement('option');
    o.value = v;
    o.textContent = v.length > 60 ? v.slice(0, 57) + '…' : v;
    sel.appendChild(o);
  }
}

// ----- download -----
async function downloadResults(format) {
  if (!lastRows.length) return;
  if (format === 'json') {
    const blob = new Blob([JSON.stringify({ rows: lastRows, columns: lastColumns }, null, 2)],
      { type: 'application/json' });
    triggerDownload(blob, 'ml-model-ranking.json');
    return;
  }
  // csv: re-post to /api/rank/csv
  const fileInput = $('#file');
  if (!fileInput.files.length) return;
  const enrich = $('#enrich').checked;
  const fd = new FormData();
  fd.append('file', fileInput.files[0]);
  fd.append('enrich', enrich ? 'true' : 'false');
  fd.append('weights', JSON.stringify(readWeights()));
  const r = await fetch('/api/rank/csv', { method: 'POST', body: fd });
  if (!r.ok) { setStatus('Pobranie CSV nie powiodło się.', 'error'); return; }
  const blob = await r.blob();
  triggerDownload(blob, 'ml-model-ranking.csv');
}
function triggerDownload(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// ----- init -----
async function init() {
  try {
    const cfg = await fetchJSON('/api/weights');
    DEFAULT_WEIGHTS = cfg.defaults || cfg.weights || cfg;
  } catch (_) {
    DEFAULT_WEIGHTS = { popularity: 0.18, architecture: 0.10, size: 0.12, accuracy: 0.25,
                        speed: 0.12, documentation: 0.08, recency: 0.08, license: 0.07 };
  }
  try {
    const p = await fetchJSON('/api/presets');
    PRESETS = (p && p.presets) ? p.presets : p || {};
  } catch (_) { PRESETS = {}; }

  // restore from localStorage
  let restored = null;
  try {
    const raw = localStorage.getItem(LS_WEIGHTS);
    if (raw) restored = JSON.parse(raw);
  } catch (_) {}
  const initial = restored && typeof restored === 'object'
    ? { ...DEFAULT_WEIGHTS, ...restored }
    : DEFAULT_WEIGHTS;
  renderWeights(initial);
  fillPresetSelect();

  try {
    const savedPreset = localStorage.getItem(LS_PRESET);
    if (savedPreset) $('#preset').value = savedPreset;
  } catch (_) {}
  try {
    const e = localStorage.getItem(LS_ENRICH);
    if (e !== null) $('#enrich').checked = e === '1';
  } catch (_) {}

  $('#run').addEventListener('click', runRanking);
  $('#download-csv').addEventListener('click', () => downloadResults('csv'));
  $('#download-json').addEventListener('click', () => downloadResults('json'));
  $('#search').addEventListener('input', () => renderResults());
  $('#modality-filter').addEventListener('change', () => renderResults());
  $('#show-norm').addEventListener('change', () => renderResults());

  $('#modal-close').addEventListener('click', closeModal);
  $('#modal .modal-backdrop').addEventListener('click', closeModal);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeModal(); });
}

init();
