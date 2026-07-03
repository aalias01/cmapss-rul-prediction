// DWG AA-101 · Turbofan RUL frontend.
// API wiring (endpoint, payload, response parsing) is unchanged from the
// previous issue of this page. Presentation follows design_language_spec.md.

const API_BASE = "https://cmapss-rul-api.onrender.com";

const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// ── DOM refs ──
const uploadZone = document.getElementById("uploadZone");
const fileInput  = document.getElementById("fileInput");
const fileInfo   = document.getElementById("fileInfo");
const fileName   = document.getElementById("fileName");
const fileRows   = document.getElementById("fileRows");
const predictBtn = document.getElementById("predictBtn");
const loadSample = document.getElementById("loadSample");

const tracesEmpty  = document.getElementById("tracesEmpty");
const tracesEl     = document.getElementById("traces");
const loadingState = document.getElementById("loadingState");
const resultState  = document.getElementById("resultState");
const errorBox     = document.getElementById("errorBox");
const errorText    = document.getElementById("errorText");

const rulValue   = document.getElementById("rulValue");
const rulBand    = document.getElementById("rulBand");
const shapBars   = document.getElementById("shapBars");
const warningBox = document.getElementById("warningBox");
const warningText = document.getElementById("warningText");

const pendingElapsed = document.getElementById("pendingElapsed");
const pendingLine    = document.getElementById("pendingLine");

// ── Theme switch (same approach as portfolio_site) ──
const modeBtn = document.getElementById("mode-switch");
function reflectTheme() {
  const t = document.documentElement.getAttribute("data-theme") || "light";
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.setAttribute("content", t === "dark" ? "#191a17" : "#f5f4ef");
  if (!modeBtn) return;
  modeBtn.setAttribute("aria-checked", t === "dark" ? "true" : "false");
  modeBtn.setAttribute("aria-label", t === "dark" ? "Switch to day mode" : "Switch to night mode");
}
modeBtn?.addEventListener("click", () => {
  const cur = document.documentElement.getAttribute("data-theme") || "light";
  const next = cur === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  try { localStorage.setItem("theme", next); } catch (e) {}
  reflectTheme();
});
reflectTheme();

// ── State ──
let parsedReadings = null;

// ── Upload zone ──
uploadZone.addEventListener("click", () => fileInput.click());
uploadZone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});
uploadZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadZone.classList.add("drag-over");
});
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("drag-over"));
uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

// ── File handling (parse logic unchanged) ──
function handleFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    const csv = e.target.result;
    const rows = parseCSV(csv);
    if (!rows || rows.length === 0) {
      showError("Could not parse that CSV. Expected a header row plus one row per cycle.");
      return;
    }
    setReadings(rows, file.name);
  };
  reader.readAsText(file);
}

function parseCSV(text) {
  const lines = text.trim().split("\n");
  if (lines.length < 2) return null;
  const headers = lines[0].split(",").map((h) => h.trim());
  return lines.slice(1).map((line) => {
    const vals = line.split(",").map((v) => v.trim());
    const obj = {};
    headers.forEach((h, i) => {
      obj[h] = isNaN(vals[i]) ? vals[i] : Number(vals[i]);
    });
    return obj;
  });
}

function setReadings(rows, name) {
  parsedReadings = rows;
  fileName.textContent = name;
  fileRows.textContent = `${rows.length} cycles`;
  fileInfo.classList.remove("hidden");
  predictBtn.disabled = false;
  hideError();
  drawTraces(rows);
}

// ── Specimen engine (generation logic unchanged; labeled synthetic on the page) ──
loadSample.addEventListener("click", (e) => {
  e.preventDefault();
  const sample = Array.from({ length: 50 }, (_, i) => {
    const deg = i / 49;
    return {
      cycle:         i + 1,
      op_setting_1:  -0.0007,
      op_setting_2:  -0.0004,
      op_setting_3:  100.0,
      sensor_2:      641.82 + deg * 2.1 + (Math.random() - 0.5) * 0.4,
      sensor_3:      1589.70 + deg * 8.5 + (Math.random() - 0.5) * 1.2,
      sensor_4:      1400.60 + deg * 6.2 + (Math.random() - 0.5) * 1.0,
      sensor_7:      554.36 - deg * 1.8 + (Math.random() - 0.5) * 0.3,
      sensor_8:      2388.06 - deg * 5.0 + (Math.random() - 0.5) * 2.0,
      sensor_9:      9046.19 - deg * 12.0 + (Math.random() - 0.5) * 3.0,
      sensor_11:     47.47 + deg * 0.4 + (Math.random() - 0.5) * 0.05,
      sensor_12:     521.66 + deg * 3.0 + (Math.random() - 0.5) * 0.5,
      sensor_13:     2388.06 - deg * 4.0 + (Math.random() - 0.5) * 1.5,
      sensor_14:     8138.62 - deg * 20.0 + (Math.random() - 0.5) * 5.0,
      sensor_15:     8.4195,
      sensor_17:     392,
      sensor_20:     39.06,
      sensor_21:     23.4190,
    };
  });
  setReadings(sample, "specimen engine (synthetic)");
});

// ── Strip traces: draw the uploaded data as the object ──
const TRACE_PREFERRED = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11", "sensor_14"];

function drawTraces(rows) {
  const keys = Object.keys(rows[0]).filter((k) => /^sensor_\d+$/.test(k));
  let chosen = TRACE_PREFERRED.filter((k) => keys.includes(k));
  if (chosen.length === 0) chosen = keys.slice(0, 6);
  chosen = chosen.slice(0, 6);
  if (chosen.length === 0) { tracesEl.classList.add("hidden"); tracesEmpty.classList.remove("hidden"); return; }

  tracesEl.innerHTML = "";
  const W = 400, H = 26, PAD = 2;

  chosen.forEach((key) => {
    const vals = rows.map((r) => Number(r[key])).filter((v) => !isNaN(v));
    if (vals.length < 2) return;
    const min = Math.min(...vals), max = Math.max(...vals);
    const span = max - min || 1;
    const pts = vals
      .map((v, i) => {
        const x = (i / (vals.length - 1)) * W;
        const y = H - PAD - ((v - min) / span) * (H - PAD * 2);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");

    const row = document.createElement("div");
    row.className = "trace-row";
    row.dataset.sensor = key;

    const name = document.createElement("span");
    name.className = "trace-name";
    name.textContent = key;

    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "trace-svg");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("preserveAspectRatio", "none");
    svg.setAttribute("aria-hidden", "true");
    const poly = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    poly.setAttribute("points", pts);
    svg.appendChild(poly);

    const range = document.createElement("span");
    range.className = "trace-range";
    range.textContent = `${fmt(min)}..${fmt(max)}`;

    row.append(name, svg, range);
    tracesEl.appendChild(row);
  });

  const axis = document.createElement("div");
  axis.className = "traces-axis";
  axis.innerHTML = `<span>CYCLE 1</span><span>NORMALIZED PER TRACE</span><span>CYCLE ${rows.length}</span>`;
  tracesEl.appendChild(axis);

  tracesEmpty.classList.add("hidden");
  tracesEl.classList.remove("hidden");
}

function fmt(v) {
  return Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2);
}

// ── Issue-pending timer (measures real elapsed time, nothing else) ──
let pendingTimer = null;
let pendingStart = 0;

function startPending() {
  pendingStart = performance.now();
  pendingElapsed.textContent = "ELAPSED 0 s";
  pendingLine.style.width = "0%";
  loadingState.classList.remove("hidden");
  pendingTimer = setInterval(() => {
    const s = Math.round((performance.now() - pendingStart) / 1000);
    pendingElapsed.textContent = `ELAPSED ${s} s`;
    // The line spans the stated 60 s worst case; it measures time, not progress.
    pendingLine.style.width = `${Math.min((s / 60) * 100, 100)}%`;
  }, 1000);
}

function stopPending() {
  clearInterval(pendingTimer);
  pendingTimer = null;
  loadingState.classList.add("hidden");
}

// ── Errors ──
function showError(msg) {
  errorText.textContent = msg;
  errorBox.classList.remove("hidden");
}
function hideError() {
  errorBox.classList.add("hidden");
}

// ── Prediction (API contract unchanged) ──
predictBtn.addEventListener("click", runPrediction);

async function runPrediction() {
  if (!parsedReadings) return;

  hideError();
  resultState.classList.add("hidden");
  startPending();
  predictBtn.disabled = true;

  const payload = { readings: parsedReadings };

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Prediction failed");
    }

    const data = await res.json();
    stopPending();
    renderResult(data);
  } catch (err) {
    stopPending();
    const friendly = /Failed to fetch|NetworkError|aborted/i.test(err.message || "")
      ? "Could not reach the API. If this was the first request in a while, the container may still be booting; wait a moment and run the estimate again."
      : err.message;
    showError(friendly);
  } finally {
    predictBtn.disabled = false;
  }
}

// ── Render result ──
function renderResult(data) {
  const rul = data.predicted_rul;

  rulValue.textContent = rul;
  rulBand.textContent = `BAND ${data.confidence_band.low} .. ${data.confidence_band.high} CYCLES`;

  // Attribution: deflections from a zero datum
  shapBars.innerHTML = "";
  const factors = data.top_factors || [];
  const maxAbs = Math.max(...factors.map((f) => Math.abs(f.value)), 0.001);

  factors.forEach((factor) => {
    const isPos = factor.value > 0;
    const widthPct = (Math.abs(factor.value) / maxAbs) * 50; // half-track per side

    const row = document.createElement("div");
    row.className = "defl-row";
    const sensorMatch = String(factor.feature).match(/^(sensor_\d+)/);
    if (sensorMatch) row.dataset.sensor = sensorMatch[1];

    const name = document.createElement("span");
    name.className = "defl-name";
    name.textContent = factor.feature;
    name.title = "SHAP value from the API response";

    const track = document.createElement("div");
    track.className = "defl-track";
    const datum = document.createElement("span");
    datum.className = "defl-datum";
    const bar = document.createElement("span");
    bar.className = `defl-bar ${isPos ? "pos" : "neg"}`;
    bar.style.width = REDUCED_MOTION ? `${widthPct}%` : "0%";
    track.append(datum, bar);

    const val = document.createElement("span");
    val.className = "defl-val";
    val.textContent = `${isPos ? "+" : ""}${factor.value.toFixed(3)}`;

    row.append(name, track, val);

    // Hover a row: trace the sensor in the principal view
    row.addEventListener("mouseenter", () => highlightTrace(row.dataset.sensor, true));
    row.addEventListener("mouseleave", () => highlightTrace(row.dataset.sensor, false));

    shapBars.appendChild(row);

    if (!REDUCED_MOTION) {
      requestAnimationFrame(() => { bar.style.transition = "width 0.4s ease"; bar.style.width = `${widthPct}%`; });
    }
  });

  // Warning from the API, rendered as a red flag note
  if (data.warning) {
    warningText.textContent = data.warning;
    warningBox.classList.remove("hidden");
  } else {
    warningBox.classList.add("hidden");
  }

  resultState.classList.remove("hidden");
}

function highlightTrace(sensor, on) {
  if (!sensor) return;
  document.querySelectorAll(`.trace-row[data-sensor="${sensor}"]`).forEach((el) => {
    el.classList.toggle("hl", on);
  });
}
