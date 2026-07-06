// AA-2026-01 · Remaining useful life (turbofan). Telemetry bench.
// API wiring (endpoint, payload, response parsing) is unchanged from the
// prior issue: the hardcoded production constant below is the contract.

const API_BASE = "https://cmapss-rul-api.onrender.com";

const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// ── DOM refs ──
const runKnownBtn = document.getElementById("runKnown");
const uploadBtn = document.getElementById("uploadBtn");
const fileInput = document.getElementById("fileInput");
const logEl = document.getElementById("log");
const tracesEl = document.getElementById("traces");

const rulValue = document.getElementById("rulValue");
const rulUnit = document.getElementById("rulUnit");
const rulRange = document.getElementById("rulRange");
const scaleEl = document.getElementById("scale");
const truthLine = document.getElementById("truthLine");
const errorBox = document.getElementById("errorBox");
const errorText = document.getElementById("errorText");
const warnBox = document.getElementById("warnBox");
const warnText = document.getElementById("warnText");
const deflectionEl = document.getElementById("deflection");
const deflRows = document.getElementById("deflRows");

const statusLine = document.getElementById("statusLine");
const nameplateWarmupEl = document.getElementById("nameplateWarmup");
const readoutWarmupEl = document.getElementById("readoutWarmup");
const modeToggle = document.getElementById("modeToggle");

const SVGNS = "http://www.w3.org/2000/svg";

const WARMUP_SECONDS = 60;
const WARMUP_GRACE_MS = 2500;
const FREE_TIER_TEXT =
  "This runs on a free tier that sleeps between visitors. First start takes 30 to 60 seconds; runs after that are quick.";
const WARMUP_COPY = {
  sent: "> server was asleep · sent the wake call",
  counting: "> warm-up estimate counting · this is an estimate, not progress",
  estimateLabel: "estimated seconds to warm",
  readyLabel: "ready",
  awake: (seconds) => `> awake · measured wake time ${seconds} s`,
  overrunLabel: "seconds elapsed · still starting",
  overrun: "> past the usual window · still waiting, counting up honestly",
};

// ── Mode toggle (shared mode cookie, localStorage fallback) ──
function currentTheme() {
  return document.documentElement.getAttribute("data-theme") === "night" ? "night" : "day";
}
function applyTheme(theme) {
  const night = theme === "night";
  document.documentElement.setAttribute("data-theme", theme);
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.setAttribute("content", night ? "#201f1c" : "#f5f4ef");
  modeToggle.setAttribute("aria-checked", night ? "true" : "false");
  modeToggle.setAttribute("aria-label", night ? "Switch to light mode" : "Switch to dark mode");
}
function persistMode(theme) {
  const mode = theme === "night" ? "dark" : "light";
  try { document.cookie = `mode=${mode}; domain=.alvinalias.com; path=/; max-age=31536000; SameSite=Lax`; } catch (e) {}
  try { localStorage.setItem("mode", mode); } catch (e) {}
}
modeToggle.addEventListener("click", () => {
  const next = currentTheme() === "night" ? "day" : "night";
  applyTheme(next);
  persistMode(next);
});
applyTheme(currentTheme());

// ── Wake-on-load: ping /health, report state in plain words ──
async function pingHealth() {
  const wake = createWarmupController({
    slot: nameplateWarmupEl,
    compact: true,
    onShow: () => { statusLine.textContent = "model waking, first run may take up to a minute"; },
  });
  try {
    const res = await fetch(`${API_BASE}/health`, { method: "GET" });
    if (res.ok) {
      const j = await res.json().catch(() => ({}));
      wake.ready();
      statusLine.textContent = j.model_loaded === true ? "model awake" : "server reachable";
    } else {
      wake.cancel();
      statusLine.textContent = "server unreachable right now";
    }
  } catch (e) {
    wake.cancel();
    statusLine.textContent = "server unreachable right now";
  }
}

// ── Sample engine manifest and rotation (random without repeat) ──
let engines = [];
let order = [];
let ptr = 0;
let lastIdx = null;

async function loadManifest() {
  try {
    const res = await fetch("samples/AA-01_engines_manifest.json");
    const j = await res.json();
    engines = j.engines || [];
    reshuffle();
  } catch (e) {
    engines = [];
  }
}
function reshuffle() {
  order = engines.map((_, i) => i);
  for (let i = order.length - 1; i > 0; i--) {
    const k = Math.floor(Math.random() * (i + 1));
    [order[i], order[k]] = [order[k], order[i]];
  }
  if (lastIdx != null && order.length > 1 && order[0] === lastIdx) {
    [order[0], order[1]] = [order[1], order[0]];
  }
  ptr = 0;
}
function nextEngine() {
  if (engines.length === 0) return null;
  if (ptr >= order.length) reshuffle();
  const idx = order[ptr++];
  lastIdx = idx;
  return engines[idx];
}

// ── Feature glosses (from the project copy deck) ──
const FEATURE_BASE = {
  sensor_3: "HPC outlet temperature",
  sensor_2: "LPC outlet temperature",
  sensor_11: "static pressure ratio",
  sensor_9: "core speed",
  sensor_14: "core speed",
};
function gloss(feature) {
  const m = String(feature).match(/^(sensor_\d+)(_mean30|_std30)?$/);
  if (!m) return feature;
  const base = FEATURE_BASE[m[1]] || m[1];
  if (m[2] === "_mean30") return `${base}, 30-cycle mean`;
  if (m[2] === "_std30") return `${base}, 30-cycle spread`;
  return base;
}

// ── CSV parse ──
function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return null;
  const headers = lines[0].split(",").map((h) => h.trim());
  const rows = lines.slice(1).map((line) => {
    const vals = line.split(",").map((v) => v.trim());
    const obj = {};
    headers.forEach((h, i) => {
      const n = Number(vals[i]);
      obj[h] = vals[i] === "" || isNaN(n) ? vals[i] : n;
    });
    return obj;
  });
  return rows.filter((r) => Object.keys(r).length > 0);
}

// ── Run log ──
let activeRunWarmup = null;
function clearLog() {
  logEl.innerHTML = "";
  dismissRunWarmup();
}
function addLog(text, cls) {
  const span = document.createElement("span");
  span.className = "log-line" + (cls ? " " + cls : "");
  span.textContent = text;
  logEl.appendChild(span);
  return span;
}

// ── Warm-up meter ──
function dismissRunWarmup() {
  if (activeRunWarmup) {
    activeRunWarmup.cancel();
    activeRunWarmup = null;
  }
}

function formatWakeSeconds(ms) {
  const seconds = ms / 1000;
  return seconds >= 10 ? String(Math.round(seconds)) : seconds.toFixed(1);
}

function buildWarmupView(slot, compact) {
  const node = document.createElement("div");
  node.className = `warmup-meter ${compact ? "warmup-compact" : "warmup-full"}`;
  node.setAttribute("role", "status");
  node.setAttribute("aria-live", "polite");

  const head = document.createElement("div");
  head.className = "warmup-head";

  const value = document.createElement("span");
  value.className = "warmup-value mono";

  const label = document.createElement("span");
  label.className = "warmup-label";

  head.append(value, label);

  const scale = document.createElement("div");
  scale.className = "warmup-scale";
  scale.setAttribute("role", "img");

  const svg = document.createElementNS(SVGNS, "svg");
  const W = 600, H = compact ? 42 : 56, baseY = compact ? 20 : 28;
  const padL = 8, padR = 8;
  const labels = [];
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("preserveAspectRatio", "none");
  svg.setAttribute("aria-hidden", "true");

  const x = (seconds) => padL + (seconds / WARMUP_SECONDS) * (W - padL - padR);
  const labelLeft = (seconds) => `${(x(seconds) / W) * 100}%`;
  const mk = (tag, attrs, cls) => {
    const el = document.createElementNS(SVGNS, tag);
    if (cls) el.setAttribute("class", cls);
    for (const k in attrs) el.setAttribute(k, attrs[k]);
    return el;
  };

  svg.appendChild(
    mk("line", { x1: x(0), y1: baseY, x2: x(WARMUP_SECONDS), y2: baseY }, "wm-base"),
  );

  for (let s = 0; s <= WARMUP_SECONDS; s += 5) {
    const major = s % 15 === 0;
    svg.appendChild(
      mk("line", {
        x1: x(s), y1: baseY, x2: x(s), y2: baseY + (major ? 10 : 6),
      }, major ? "wm-tick-major" : "wm-tick-minor"),
    );
    if (major) {
      labels.push({ text: String(s), left: labelLeft(s) });
    }
  }

  const marker = mk(
    "polygon",
    { points: `0,${baseY - 2} -6,${baseY - 14} 6,${baseY - 14}` },
    "wm-marker",
  );
  svg.appendChild(marker);
  scale.appendChild(svg);
  const labelLayer = document.createElement("div");
  labelLayer.className = "warmup-scale-labels";
  labelLayer.setAttribute("aria-hidden", "true");
  labels.forEach((item) => {
    const span = document.createElement("span");
    span.className = "wm-num-label";
    span.style.left = item.left;
    span.textContent = item.text;
    labelLayer.appendChild(span);
  });
  scale.appendChild(labelLayer);

  const log = document.createElement("div");
  log.className = "warmup-log mono";

  const note = document.createElement("p");
  note.className = "warmup-note";
  note.textContent = FREE_TIER_TEXT;
  if (compact) note.hidden = true;

  node.append(head, scale, log, note);
  slot.innerHTML = "";
  slot.appendChild(node);
  slot.classList.remove("hidden");

  let markerSeconds = WARMUP_SECONDS;
  let markerAnim = null;

  function markerX(seconds) {
    return x(Math.max(0, Math.min(WARMUP_SECONDS, seconds)));
  }
  function setMarker(seconds, overrun, animate) {
    const from = markerX(markerSeconds);
    const to = markerX(seconds);
    markerSeconds = seconds;
    marker.classList.toggle("wm-marker-red", overrun);
    if (markerAnim) cancelAnimationFrame(markerAnim);
    if (!animate || REDUCED_MOTION) {
      marker.setAttribute("transform", `translate(${to} 0)`);
      return;
    }
    const started = performance.now();
    const step = (now) => {
      const pct = Math.min(1, (now - started) / 420);
      const eased = 1 - Math.pow(1 - pct, 3);
      marker.setAttribute("transform", `translate(${from + (to - from) * eased} 0)`);
      if (pct < 1) markerAnim = requestAnimationFrame(step);
    };
    markerAnim = requestAnimationFrame(step);
  }

  function setLogs(lines) {
    log.innerHTML = "";
    lines.forEach((line) => {
      const span = document.createElement("span");
      span.className = "warmup-log-line";
      span.textContent = line;
      log.appendChild(span);
    });
  }

  function update(state) {
    value.textContent = state.value;
    label.textContent = state.label;
    setLogs(state.logs);
    setMarker(state.markerSeconds, state.overrun, state.animate);
    scale.setAttribute(
      "aria-label",
      `${state.value} ${state.label}. Scale from 0 to 60 seconds.`,
    );
  }

  function remove() {
    if (markerAnim) cancelAnimationFrame(markerAnim);
    slot.innerHTML = "";
    slot.classList.add("hidden");
  }

  return { node, update, remove };
}

function createWarmupController(opts) {
  const started = performance.now();
  const slot = opts.slot;
  const compact = opts.compact === true;
  let view = null;
  let done = false;
  let overrunLogged = false;
  let tickTimer = null;
  let graceTimer = null;
  let retireTimer = null;
  let interactionCleanup = null;

  function elapsedMs() {
    return performance.now() - started;
  }

  function updateCounting() {
    if (!view || done) return;
    const elapsed = Math.floor(elapsedMs() / 1000);
    if (elapsed >= WARMUP_SECONDS) {
      if (!overrunLogged) overrunLogged = true;
      view.update({
        value: String(elapsed),
        label: WARMUP_COPY.overrunLabel,
        markerSeconds: 0,
        overrun: true,
        animate: false,
        logs: [WARMUP_COPY.sent, WARMUP_COPY.overrun],
      });
      return;
    }
    const remaining = Math.max(0, WARMUP_SECONDS - elapsed);
    view.update({
      value: String(remaining),
      label: WARMUP_COPY.estimateLabel,
      markerSeconds: remaining,
      overrun: false,
      animate: false,
      logs: [WARMUP_COPY.sent, WARMUP_COPY.counting],
    });
  }

  function show() {
    if (done || view) return;
    view = buildWarmupView(slot, compact);
    if (opts.onShow) opts.onShow();
    updateCounting();
    tickTimer = setInterval(updateCounting, 1000);
  }

  graceTimer = setTimeout(show, WARMUP_GRACE_MS);

  function clearTimers() {
    if (graceTimer) { clearTimeout(graceTimer); graceTimer = null; }
    if (tickTimer) { clearInterval(tickTimer); tickTimer = null; }
    if (retireTimer) { clearTimeout(retireTimer); retireTimer = null; }
    if (interactionCleanup) { interactionCleanup(); interactionCleanup = null; }
  }

  function retire() {
    if (!view) return;
    clearTimers();
    if (REDUCED_MOTION) {
      view.remove();
      view = null;
      return;
    }
    view.node.classList.add("warmup-retiring");
    retireTimer = setTimeout(() => {
      if (view) {
        view.remove();
        view = null;
      }
    }, 280);
  }

  function armRetirement() {
    if (!view) return;
    if (REDUCED_MOTION) {
      retire();
      return;
    }
    const interaction = () => retire();
    document.addEventListener("pointerdown", interaction, { once: true });
    document.addEventListener("keydown", interaction, { once: true });
    interactionCleanup = () => {
      document.removeEventListener("pointerdown", interaction);
      document.removeEventListener("keydown", interaction);
    };
    retireTimer = setTimeout(retire, 4000);
  }

  return {
    ready() {
      if (done) return null;
      done = true;
      if (graceTimer) { clearTimeout(graceTimer); graceTimer = null; }
      if (tickTimer) { clearInterval(tickTimer); tickTimer = null; }
      const measured = formatWakeSeconds(elapsedMs());
      const line = WARMUP_COPY.awake(measured);
      if (view) {
        view.update({
          value: "0",
          label: WARMUP_COPY.readyLabel,
          markerSeconds: 0,
          overrun: false,
          animate: true,
          logs: [line],
        });
        armRetirement();
        return line;
      }
      return null;
    },
    cancel() {
      done = true;
      clearTimers();
      if (view) {
        view.remove();
        view = null;
      }
    },
  };
}

// ── Traces (sensors 3, 2, 11 per the dialect) ──
const TRACE_SENSORS = [
  { key: "sensor_3", n: 3, gloss: "HPC outlet temperature" },
  { key: "sensor_2", n: 2, gloss: "LPC outlet temperature" },
  { key: "sensor_11", n: 11, gloss: "static pressure ratio" },
];
function drawTraces(rows) {
  tracesEl.innerHTML = "";
  const W = 500, H = 30, PAD = 3;
  TRACE_SENSORS.forEach((s) => {
    const vals = rows.map((r) => Number(r[s.key])).filter((v) => !isNaN(v));
    if (vals.length < 2) return;
    const min = Math.min(...vals), max = Math.max(...vals), span = (max - min) || 1;
    const pts = vals
      .map((v, i) => {
        const x = PAD + (i / (vals.length - 1)) * (W - 2 * PAD);
        const y = H - PAD - ((v - min) / span) * (H - 2 * PAD);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");

    const row = document.createElement("div");
    row.className = "trace-row";
    row.dataset.sensor = s.key;

    const label = document.createElement("span");
    label.className = "trace-label";
    label.textContent = `sensor ${s.n} · ${s.gloss} · last ${vals.length} cycles`;

    const svg = document.createElementNS(SVGNS, "svg");
    svg.setAttribute("class", "trace-svg");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("preserveAspectRatio", "none");
    svg.setAttribute("aria-hidden", "true");

    const base = document.createElementNS(SVGNS, "line");
    base.setAttribute("class", "trace-base");
    base.setAttribute("x1", PAD); base.setAttribute("y1", H - PAD);
    base.setAttribute("x2", W - PAD); base.setAttribute("y2", H - PAD);

    const poly = document.createElementNS(SVGNS, "polyline");
    poly.setAttribute("points", pts);

    svg.append(base, poly);
    row.append(label, svg);
    tracesEl.appendChild(row);
  });
}

// ── House scale: 0 to 125 cycles, major every 25, minor every 5 ──
function drawScale(opts) {
  const o = opts || {};
  const W = 500, H = 64, padL = 8, padR = 8;
  const baseY = 42;
  const x = (v) => padL + (v / 125) * (W - padL - padR);
  const labels = [];
  const labelLeft = (v) => `${(x(v) / W) * 100}%`;

  const svg = document.createElementNS(SVGNS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("preserveAspectRatio", "none");
  svg.setAttribute("aria-hidden", "true");

  const mk = (tag, attrs, cls) => {
    const el = document.createElementNS(SVGNS, tag);
    if (cls) el.setAttribute("class", cls);
    for (const k in attrs) el.setAttribute(k, attrs[k]);
    return el;
  };

  svg.appendChild(mk("line", { x1: x(0), y1: baseY, x2: x(125), y2: baseY }, "sc-base"));

  for (let v = 0; v <= 125; v += 5) {
    const major = v % 25 === 0;
    svg.appendChild(mk("line", {
      x1: x(v), y1: baseY, x2: x(v), y2: baseY + (major ? 8 : 4),
    }, major ? "sc-tick-major" : "sc-tick-minor"));
    if (major) {
      labels.push({ text: String(v), left: labelLeft(v), cls: "sc-num-label" });
    }
  }

  if (typeof o.rul === "number") {
    // band bracket above the scale
    if (typeof o.low === "number" && typeof o.high === "number") {
      const by = 24;
      svg.appendChild(mk("line", { x1: x(o.low), y1: by, x2: x(o.high), y2: by }, "sc-band"));
      svg.appendChild(mk("line", { x1: x(o.low), y1: by, x2: x(o.low), y2: by + 5 }, "sc-band"));
      svg.appendChild(mk("line", { x1: x(o.high), y1: by, x2: x(o.high), y2: by + 5 }, "sc-band"));
    }
    // truth: dashed line + label
    if (typeof o.truth === "number") {
      svg.appendChild(mk("line", { x1: x(o.truth), y1: 16, x2: x(o.truth), y2: baseY + 8 }, "sc-truth"));
      labels.push({ text: `official ${o.truth}`, left: labelLeft(o.truth), cls: "sc-truth-label" });
    }
    // marker: filled triangle pointing down onto the baseline
    const mx = x(o.rul);
    const tri = mk("polygon", { points: `${mx},${baseY - 2} ${mx - 5},${baseY - 12} ${mx + 5},${baseY - 12}` }, "sc-marker");
    svg.appendChild(tri);
  }

  scaleEl.innerHTML = "";
  scaleEl.appendChild(svg);
  const labelLayer = document.createElement("div");
  labelLayer.className = "scale-labels";
  labelLayer.setAttribute("aria-hidden", "true");
  labels.forEach((item) => {
    const span = document.createElement("span");
    span.className = item.cls;
    span.style.left = item.left;
    span.textContent = item.text;
    labelLayer.appendChild(span);
  });
  scaleEl.appendChild(labelLayer);

  let aria = "Scale from 0 to 125 cycles.";
  if (typeof o.rul === "number") {
    aria += ` Estimate ${o.rul}, range ${o.low} to ${o.high}.`;
    if (typeof o.truth === "number") aria += ` Official answer ${o.truth}.`;
  }
  scaleEl.setAttribute("aria-label", aria);

  // Reveal the marker and band once, under 600ms.
  if (typeof o.rul === "number" && !REDUCED_MOTION) {
    svg.querySelectorAll(".sc-marker, .sc-band").forEach((el) => { el.style.opacity = "0"; });
    requestAnimationFrame(() => {
      svg.querySelectorAll(".sc-marker, .sc-band").forEach((el) => { el.style.opacity = "1"; });
    });
  }
}

// ── Deflection table (side from the API direction field, not the sign) ──
function renderDeflection(factors) {
  deflRows.innerHTML = "";
  const list = factors || [];
  const maxAbs = Math.max(...list.map((f) => Math.abs(f.value)), 0.001);

  list.forEach((f) => {
    const right = f.direction === "increases_rul";
    const widthPct = (Math.abs(f.value) / maxAbs) * 50;

    const row = document.createElement("div");
    row.className = "defl-row";
    const base = String(f.feature).match(/^(sensor_\d+)/);
    if (base) row.dataset.sensor = base[1];

    const name = document.createElement("span");
    name.className = "defl-name";
    name.textContent = gloss(f.feature);
    name.title = `${f.feature} · SHAP value from the API response`;

    const track = document.createElement("div");
    track.className = "defl-track";
    const datum = document.createElement("span");
    datum.className = "defl-datum";
    const bar = document.createElement("span");
    bar.className = `defl-bar ${right ? "pos" : "neg"}`;
    bar.style.width = REDUCED_MOTION ? `${widthPct}%` : "0%";
    track.append(datum, bar);

    const val = document.createElement("span");
    val.className = "defl-val";
    val.textContent = `${f.value > 0 ? "+" : ""}${Number(f.value).toFixed(3)}`;

    row.append(name, track, val);
    row.addEventListener("mouseenter", () => highlightTrace(row.dataset.sensor, true));
    row.addEventListener("mouseleave", () => highlightTrace(row.dataset.sensor, false));
    deflRows.appendChild(row);

    if (!REDUCED_MOTION) {
      requestAnimationFrame(() => { bar.style.width = `${widthPct}%`; });
    }
  });
  deflectionEl.classList.remove("hidden");
}
function highlightTrace(sensor, on) {
  if (!sensor) return;
  document.querySelectorAll(`.trace-row[data-sensor="${sensor}"]`).forEach((el) => {
    el.classList.toggle("hl", on);
  });
}

// ── Errors and reset ──
function showError(msg) {
  errorText.textContent = msg;
  errorBox.classList.remove("hidden");
}
function resetReadout() {
  errorBox.classList.add("hidden");
  warnBox.classList.add("hidden");
  deflectionEl.classList.add("hidden");
  truthLine.textContent = "";
  rulValue.textContent = "";
  rulUnit.textContent = "";
  rulRange.textContent = "";
  drawScale({});
}

function setBusy(busy) {
  runKnownBtn.disabled = busy;
  uploadBtn.disabled = busy;
}

// ── Prediction (API contract unchanged) ──
async function predict(readings, meta) {
  resetReadout();
  setBusy(true);
  addLog("> sent to model");
  activeRunWarmup = createWarmupController({
    slot: readoutWarmupEl,
    compact: false,
  });
  const t0 = performance.now();

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ readings }),
    });

    const wakeLine = activeRunWarmup ? activeRunWarmup.ready() : null;
    activeRunWarmup = null;
    if (wakeLine) addLog(wakeLine);

    if (!res.ok) {
      let detail = "";
      try {
        const err = await res.json();
        detail = typeof err.detail === "string"
          ? err.detail
          : Array.isArray(err.detail) ? err.detail.map((d) => d.msg).join("; ") : "";
      } catch (e) {}
      showError(`The model could not score this file. ${detail} Check the columns against the sample CSV and try again.`);
      return;
    }

    const data = await res.json();
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    addLog(`> answered in ${secs} s`);
    renderResult(data, meta);
    statusLine.textContent = "model awake";
  } catch (e) {
    dismissRunWarmup();
    showError("Could not reach the model. If this is the first run in a while, the server is still waking; that takes 30 to 60 seconds. Try again in a moment.");
  } finally {
    setBusy(false);
  }
}

function renderResult(data, meta) {
  const rul = data.predicted_rul;
  const low = data.confidence_band.low;
  const high = data.confidence_band.high;

  rulValue.textContent = rul;
  rulUnit.textContent = "cycles remaining";
  rulRange.textContent = `range ${low} to ${high}`;

  const hasTruth = meta.kind === "known" && typeof meta.official === "number";
  drawScale({ rul, low, high, truth: hasTruth ? meta.official : undefined });

  if (hasTruth) {
    const err = rul - meta.official;
    const signed = err === 0 ? "0" : (err > 0 ? `+${err}` : `−${Math.abs(err)}`);
    truthLine.textContent = `Official answer for engine ${meta.unit}: ${meta.official} · model error ${signed} cycles`;
  } else {
    truthLine.textContent = "Uploaded data has no official answer to compare. The range still applies.";
  }

  if (data.warning) {
    warnText.textContent = data.warning;
    warnBox.classList.remove("hidden");
  }

  renderDeflection(data.top_factors);
}

// ── Run a known engine ──
runKnownBtn.addEventListener("click", async () => {
  const eng = nextEngine();
  if (!eng) {
    clearLog();
    showError("Could not reach the model. If this is the first run in a while, the server is still waking; that takes 30 to 60 seconds. Try again in a moment.");
    return;
  }
  clearLog();
  setBusy(true);
  let rows;
  try {
    const res = await fetch(`samples/${eng.file}`);
    rows = parseCSV(await res.text());
  } catch (e) {
    setBusy(false);
    showError("Could not reach the model. If this is the first run in a while, the server is still waking; that takes 30 to 60 seconds. Try again in a moment.");
    return;
  }
  if (!rows || rows.length === 0) {
    setBusy(false);
    showError("The model could not score this file. The sample engine could not be read. Check the columns against the sample CSV and try again.");
    return;
  }
  addLog(`> engine ${eng.engine_unit} loaded · ${rows.length} cycles of sensor history`);
  drawTraces(rows);
  await predict(rows, { kind: "known", unit: eng.engine_unit, official: eng.official_rul });
});

// ── Upload your own CSV ──
uploadBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    const rows = parseCSV(e.target.result);
    if (!rows || rows.length === 0) {
      clearLog();
      showError("The model could not score this file. The file had no readable rows. Check the columns against the sample CSV and try again.");
      return;
    }
    clearLog();
    addLog(`> file ${file.name} loaded · ${rows.length} cycles of sensor history`);
    drawTraces(rows);
    predict(rows, { kind: "upload", name: file.name });
  };
  reader.readAsText(file);
  fileInput.value = "";
});

// ── Init ──
drawScale({});
pingHealth();
loadManifest();
