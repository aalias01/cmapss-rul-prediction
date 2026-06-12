const API_BASE = "https://cmapss-rul-api.onrender.com";

// ─── DOM refs ────────────────────────────────────────────────────────────────
const uploadZone  = document.getElementById("uploadZone");
const fileInput   = document.getElementById("fileInput");
const fileInfo    = document.getElementById("fileInfo");
const fileName    = document.getElementById("fileName");
const fileRows    = document.getElementById("fileRows");
const predictBtn  = document.getElementById("predictBtn");
const loadSample  = document.getElementById("loadSample");

const emptyState   = document.getElementById("emptyState");
const loadingState = document.getElementById("loadingState");
const resultState  = document.getElementById("resultState");

const rulValue   = document.getElementById("rulValue");
const rulBand    = document.getElementById("rulBand");
const gaugeFill  = document.getElementById("gaugeFill");
const shapBars   = document.getElementById("shapBars");
const warningBox = document.getElementById("warningBox");
const warningText = document.getElementById("warningText");

// ─── State ───────────────────────────────────────────────────────────────────
let parsedReadings = null;

// ─── Upload zone interaction ──────────────────────────────────────────────────
uploadZone.addEventListener("click", () => fileInput.click());

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

// ─── File handling ────────────────────────────────────────────────────────────
function handleFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    const csv = e.target.result;
    const rows = parseCSV(csv);
    if (!rows || rows.length === 0) {
      alert("Could not parse CSV. Check the file format.");
      return;
    }
    parsedReadings = rows;
    fileName.textContent = file.name;
    fileRows.textContent = `${rows.length} cycles`;
    fileInfo.classList.remove("hidden");
    predictBtn.disabled = false;
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

// ─── Sample data ──────────────────────────────────────────────────────────────
loadSample.addEventListener("click", (e) => {
  e.preventDefault();
  // Realistic CMAPSS-like sensor readings for a degrading engine (50 cycles)
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
  parsedReadings = sample;
  fileName.textContent = "sample_engine.csv";
  fileRows.textContent = "50 cycles";
  fileInfo.classList.remove("hidden");
  predictBtn.disabled = false;
});

// ─── Prediction ───────────────────────────────────────────────────────────────
predictBtn.addEventListener("click", runPrediction);

async function runPrediction() {
  if (!parsedReadings) return;

  showState("loading");

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
    renderResult(data);
    showState("result");
  } catch (err) {
    showState("empty");
    alert(`Error: ${err.message}`);
  }
}

// ─── Render result ────────────────────────────────────────────────────────────
function renderResult(data) {
  const rul = data.predicted_rul;

  // RUL number + color coding
  rulValue.textContent = rul;
  rulValue.className = "rul-value";
  if (rul < 30)       rulValue.classList.add("critical");
  else if (rul < 70)  rulValue.classList.add("caution");
  else                rulValue.classList.add("healthy");

  // Confidence band
  rulBand.textContent = `Range: ${data.confidence_band.low} – ${data.confidence_band.high} cycles`;

  // Gauge
  const pct = Math.round((rul / 125) * 100);
  gaugeFill.style.width = `${pct}%`;

  // SHAP bars
  shapBars.innerHTML = "";
  const maxAbs = Math.max(...data.top_factors.map((f) => Math.abs(f.value)));

  data.top_factors.forEach((factor) => {
    const isPos = factor.value > 0;
    const barWidth = Math.round((Math.abs(factor.value) / maxAbs) * 100);
    const dirLabel = isPos ? "↑ increases RUL estimate" : "↓ decreases RUL estimate";

    const row = document.createElement("div");
    row.className = "shap-row";
    row.innerHTML = `
      <div class="shap-row-header">
        <span class="shap-feat">${factor.feature}</span>
        <span class="shap-val ${isPos ? "positive" : "negative"}">${isPos ? "+" : ""}${factor.value.toFixed(3)}</span>
      </div>
      <div class="shap-bar-track">
        <div class="shap-bar-fill ${isPos ? "positive" : "negative"}" style="width:${barWidth}%"></div>
      </div>
      <div class="shap-dir">${dirLabel}</div>
    `;
    shapBars.appendChild(row);
  });

  // Warning
  if (data.warning) {
    warningText.textContent = data.warning;
    warningBox.classList.remove("hidden");
  } else {
    warningBox.classList.add("hidden");
  }
}

// ─── State switcher ───────────────────────────────────────────────────────────
function showState(state) {
  emptyState.classList.add("hidden");
  loadingState.classList.add("hidden");
  resultState.classList.add("hidden");

  if (state === "empty")   emptyState.classList.remove("hidden");
  if (state === "loading") loadingState.classList.remove("hidden");
  if (state === "result")  resultState.classList.remove("hidden");
}
