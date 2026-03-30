const API_URL = "http://localhost:5000";

let cities      = [];
let running     = false;
let energyChart = null;
let lastResult  = null;


function initEnergyChart() {
  const ctx = document.getElementById("energy-chart").getContext("2d");
  if (energyChart) energyChart.destroy();
  energyChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [{
        data: [],
        borderColor: "#4f8ef7",
        borderWidth: 1.5,
        pointRadius: 0,
        fill: true,
        backgroundColor: "rgba(79,142,247,0.08)",
        tension: 0.3,
      }],
    },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: true,
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      scales: {
        x: { display: false },
        y: {
          display: true,
          ticks: {
            color: "#555c70",
            font: { family: "JetBrains Mono", size: 10 },
            maxTicksLimit: 4,
            callback: v => v >= 1000 ? (v / 1000).toFixed(1) + "k" : v.toFixed(0),
          },
          grid:   { color: "#1a1e2a" },
          border: { display: false },
        },
      },
    },
  });
}

function populateEnergyChart(energyHistory) {
  const d = energyChart.data;
  d.labels           = energyHistory.map((_, i) => i * 100);
  d.datasets[0].data = energyHistory;
  energyChart.update("none");
}


const cityCanvas = document.getElementById("city-canvas");
const cityCtx    = cityCanvas.getContext("2d");

function resizeCanvases() {
  const w = cityCanvas.offsetWidth;
  cityCanvas.width  = w;
  cityCanvas.height = w;
  const hm = document.getElementById("heatmap-canvas");
  hm.width  = hm.offsetWidth;
  hm.height = hm.offsetWidth;
  drawCities();
  drawHeatmap(null);
}

function toCanvas(x, y) {
  const w = cityCanvas.width, pad = 30;
  return { cx: pad + x * (w - 2 * pad), cy: pad + y * (w - 2 * pad) };
}

function drawCities(tour = null, valid = null) {
  const w = cityCanvas.width, h = cityCanvas.height, pad = 30;
  cityCtx.clearRect(0, 0, w, h);

  cityCtx.strokeStyle = "#1a1e2a";
  cityCtx.lineWidth = 0.5;
  for (let i = 0; i <= 5; i++) {
    const x = pad + i * (w - 2 * pad) / 5;
    const y = pad + i * (h - 2 * pad) / 5;
    cityCtx.beginPath(); cityCtx.moveTo(x, pad);  cityCtx.lineTo(x, h - pad); cityCtx.stroke();
    cityCtx.beginPath(); cityCtx.moveTo(pad, y);  cityCtx.lineTo(w - pad, y); cityCtx.stroke();
  }

  if (tour && cities.length > 1) {
    const color = valid ? "#34d399" : "#f87171";
    cityCtx.strokeStyle = color;
    cityCtx.lineWidth = 2;
    cityCtx.setLineDash([]);
    for (let s = 0; s < tour.length - 1; s++) {
      const a = toCanvas(cities[tour[s]].x, cities[tour[s]].y);
      const b = toCanvas(cities[tour[s + 1]].x, cities[tour[s + 1]].y);
      cityCtx.beginPath(); cityCtx.moveTo(a.cx, a.cy); cityCtx.lineTo(b.cx, b.cy); cityCtx.stroke();
      const angle = Math.atan2(b.cy - a.cy, b.cx - a.cx);
      const mx = (a.cx + b.cx) / 2, my = (a.cy + b.cy) / 2;
      cityCtx.save();
      cityCtx.translate(mx, my); cityCtx.rotate(angle);
      cityCtx.beginPath(); cityCtx.moveTo(0, 0); cityCtx.lineTo(-10, 5); cityCtx.lineTo(-10, -5); cityCtx.closePath();
      cityCtx.fillStyle = color; cityCtx.fill();
      cityCtx.restore();
    }
  }

  cities.forEach((c, i) => {
    const { cx, cy } = toCanvas(c.x, c.y);
    const grad = cityCtx.createRadialGradient(cx, cy, 0, cx, cy, 18);
    grad.addColorStop(0, "rgba(79,142,247,0.18)");
    grad.addColorStop(1, "rgba(79,142,247,0)");
    cityCtx.fillStyle = grad;
    cityCtx.beginPath(); cityCtx.arc(cx, cy, 18, 0, Math.PI * 2); cityCtx.fill();
    cityCtx.fillStyle = "#4f8ef7";
    cityCtx.beginPath(); cityCtx.arc(cx, cy, 7, 0, Math.PI * 2); cityCtx.fill();
    cityCtx.strokeStyle = "#0d0f14"; cityCtx.lineWidth = 2; cityCtx.stroke();
    cityCtx.fillStyle = "#e8eaf0";
    cityCtx.font = "bold 12px 'JetBrains Mono'";
    cityCtx.textAlign = "center"; cityCtx.textBaseline = "middle";
    cityCtx.fillText(String.fromCharCode(65 + i), cx, cy - 18);
  });
}

cityCanvas.addEventListener("click", e => {
  if (running) return;
  const r  = cityCanvas.getBoundingClientRect();
  const w  = cityCanvas.width, pad = 30;
  const px = (e.clientX - r.left) * (cityCanvas.width  / r.width);
  const py = (e.clientY - r.top)  * (cityCanvas.height / r.height);
  const x  = (px - pad) / (w - 2 * pad);
  const y  = (py - pad) / (w - 2 * pad);
  if (x < 0 || x > 1 || y < 0 || y > 1) return;
  if (cities.length >= 10) { showDiag('<span class="err">max 10 cities</span>'); return; }
  cities.push({ x, y });
  updateCityCount();
  lastResult = null;
  drawCities();
});

cityCanvas.addEventListener("contextmenu", e => {
  e.preventDefault();
  if (running) return;
  const r  = cityCanvas.getBoundingClientRect();
  const px = (e.clientX - r.left) * (cityCanvas.width  / r.width);
  const py = (e.clientY - r.top)  * (cityCanvas.height / r.height);
  let closest = -1, bestDist = Infinity;
  cities.forEach((c, i) => {
    const { cx, cy } = toCanvas(c.x, c.y);
    const d = Math.hypot(px - cx, py - cy);
    if (d < bestDist) { bestDist = d; closest = i; }
  });
  if (closest >= 0 && bestDist < 20) {
    cities.splice(closest, 1);
    updateCityCount();
    drawCities();
  }
});

function updateCityCount() {
  document.getElementById("stat-cities").textContent = cities.length;
}


function drawHeatmap(V) {
  const canvas = document.getElementById("heatmap-canvas");
  const ctx    = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const N = cities.length;

  if (!V || N === 0) {
    ctx.fillStyle = "#1a1e2a";
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = "#555c70";
    ctx.font = "11px 'JetBrains Mono'";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText("no data", w / 2, h / 2);
    return;
  }

  const cellW = w / N, cellH = h / N;
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const v = Math.max(0, Math.min(1, V[i][j]));
      const r = Math.round(v * 251 + (1 - v) * 30);
      const g = Math.round(v * 191 + (1 - v) * 40);
      const b = Math.round(v * 36  + (1 - v) * 100);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(j * cellW, i * cellH, cellW, cellH);
      if (N <= 8) {
        ctx.fillStyle = v > 0.5 ? "#0d0f14" : "#8b90a0";
        ctx.font = `${Math.min(10, cellW * 0.35)}px 'JetBrains Mono'`;
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText(v.toFixed(2), j * cellW + cellW / 2, i * cellH + cellH / 2);
      }
    }
  }

  if (N <= 8) {
    ctx.fillStyle = "rgba(13,15,20,0.7)";
    ctx.fillRect(0, 0, w, 14);
    ctx.fillRect(0, 0, 14, h);
    ctx.fillStyle = "#555c70";
    ctx.font = "9px 'JetBrains Mono'";
    ctx.textAlign = "center";
    for (let j = 0; j < N; j++) ctx.fillText("p" + (j + 1), j * cellW + cellW / 2, 9);
    ctx.textAlign = "right";
    for (let i = 0; i < N; i++) {
      ctx.textBaseline = "middle";
      ctx.fillText(String.fromCharCode(65 + i), 12, i * cellH + cellH / 2);
    }
  }
}


function getParams() {
  return {
    penaltyRow:      +document.getElementById("sl-row").value,
    penaltyCol:      +document.getElementById("sl-col").value,
    penaltyDistance: +document.getElementById("sl-dist").value,
    penaltyToursize: +document.getElementById("sl-tour").value,
    maxIter:         +document.getElementById("sl-iter").value,
  };
}

async function startRun() {
  if (cities.length < 3) { showDiag('<span class="err">place at least 3 cities first</span>'); return; }
  if (running) return;

  running = true;
  setBadge("running");
  document.getElementById("btn-run").textContent = "⏳ Solving…";
  document.getElementById("btn-run").disabled = true;
  document.getElementById("tour-display").textContent = "solving…";
  document.getElementById("tour-display").style.color = "var(--amber)";
  showDiag('<span class="info">sent to Python… running network.py</span>');
  initEnergyChart();

  try {
    const res = await fetch(`${API_URL}/solve`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ cities, ...getParams() }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || `HTTP ${res.status}`);
    }

    handleResult(await res.json());

  } catch (err) {
    setBadge("idle");
    showDiag(
      `<span class="err">could not reach server</span>\n` +
      `make sure server.py is running:\n\n  python server.py`
    );
    document.getElementById("tour-display").textContent = "no result yet";
    document.getElementById("tour-display").style.color = "var(--text3)";
  } finally {
    running = false;
    document.getElementById("btn-run").textContent = "▶ Run";
    document.getElementById("btn-run").disabled = false;
  }
}

function handleResult(data) {
  if (data.energyHistory?.length) populateEnergyChart(data.energyHistory);
  if (data.activation) drawHeatmap(data.activation);

  document.getElementById("stat-iter").textContent =
    data.energyHistory ? (data.energyHistory.length * 100).toLocaleString() : "—";

  if (data.valid) {
    setBadge("valid");
    drawCities(data.tour, true);
    document.getElementById("tour-display").textContent = data.tourLabels.join(" → ");
    document.getElementById("tour-display").style.color = "var(--green)";
    document.getElementById("stat-dist").textContent    = data.tourDistance.toFixed(3);
    document.getElementById("stat-dist").className      = "stat-val green";
    const pct = data.twoOptImprovement;
    document.getElementById("stat-2opt").textContent    = pct > 0 ? `-${pct.toFixed(1)}%` : "0%";
    document.getElementById("stat-2opt").className      = pct > 0 ? "stat-val green" : "stat-val";
    document.getElementById("stat-trials").textContent  = `${data.trialsRun}`;
    showDiag(
      `<span class="ok">✓ valid tour found</span>\n` +
      `distance: ${data.tourDistance.toFixed(4)}\n` +
      (pct > 0 ? `2-opt improved by ${pct.toFixed(1)}%\n` : "") +
      `${data.tour.length - 1} edges · best of ${data.trialsRun} trials`
    );
  } else {
    setBadge("invalid");
    drawCities(null, false);
    document.getElementById("tour-display").textContent = "invalid — see diagnostics";
    document.getElementById("tour-display").style.color = "var(--red)";
    document.getElementById("stat-dist").textContent    = "—";
    document.getElementById("stat-dist").className      = "stat-val";
    document.getElementById("stat-2opt").textContent    = "—";
    document.getElementById("stat-2opt").className      = "stat-val";
    document.getElementById("stat-trials").textContent  = `${data.trialsRun}`;
    const msgs = data.diagnostics?.length ? data.diagnostics : ["no diagnostics returned"];
    showDiag(`<span class="err">✗ no valid tour</span>\n` + msgs.map(m => "· " + m).join("\n"));
  }

  lastResult = data;
}


function seededRandom(seed) {
  let s = seed % 2147483647;
  if (s <= 0) s += 2147483646;
  return () => { s = s * 16807 % 2147483647; return (s - 1) / 2147483646; };
}

function randomizeCities() {
  if (running) return;
  const rng = seededRandom(Date.now() % 9999);
  cities = Array.from({ length: 6 }, () => ({ x: 0.1 + rng() * 0.8, y: 0.1 + rng() * 0.8 }));
  lastResult = null;
  updateCityCount(); drawCities(); drawHeatmap(null);
  setBadge("idle");
  document.getElementById("tour-display").textContent = "no result yet";
  document.getElementById("tour-display").style.color = "var(--text3)";
  document.getElementById("stat-dist").textContent    = "—";
  document.getElementById("stat-dist").className      = "stat-val";
  showDiag("cities randomised. press ▶ Run.");
}

function clearCities() {
  if (running) return;
  cities = []; lastResult = null;
  updateCityCount(); drawCities(); drawHeatmap(null);
  setBadge("idle");
  document.getElementById("tour-display").textContent = "no result yet";
  document.getElementById("tour-display").style.color = "var(--text3)";
  document.getElementById("stat-iter").textContent    = "—";
  document.getElementById("stat-energy").textContent  = "—";
  document.getElementById("stat-dist").textContent    = "—";
  document.getElementById("stat-dist").className      = "stat-val";
  document.getElementById("stat-2opt").textContent    = "—";
  document.getElementById("stat-2opt").className      = "stat-val";
  document.getElementById("stat-trials").textContent  = "—";
  document.getElementById("progress-bar").style.width = "0%";
  showDiag("place 3–8 cities, tune penalties, then run.");
  initEnergyChart();
}


function setBadge(state) {
  const el = document.getElementById("status-badge");
  el.className   = "badge badge-" + state;
  el.textContent = state.toUpperCase();
}

function showDiag(html) {
  document.getElementById("diag-text").innerHTML = html;
}

function applyPreset(row, col, dist, tour) {
  document.getElementById("sl-row").value  = row;  document.getElementById("v-row").textContent  = row;
  document.getElementById("sl-col").value  = col;  document.getElementById("v-col").textContent  = col;
  document.getElementById("sl-dist").value = dist; document.getElementById("v-dist").textContent = dist;
  document.getElementById("sl-tour").value = tour; document.getElementById("v-tour").textContent = tour;
}


document.getElementById("btn-run").addEventListener("click", startRun);
document.getElementById("btn-randomize").addEventListener("click", randomizeCities);
document.getElementById("btn-clear").addEventListener("click", clearCities);

[
  ["sl-row",  "v-row",  v => v],
  ["sl-col",  "v-col",  v => v],
  ["sl-dist", "v-dist", v => v],
  ["sl-tour", "v-tour", v => v],
  ["sl-iter", "v-iter", v => parseInt(v).toLocaleString()],
].forEach(([sliderId, labelId, fmt]) => {
  document.getElementById(sliderId).addEventListener("input", function () {
    document.getElementById(labelId).textContent = fmt(this.value);
  });
});

document.querySelectorAll(".speed-btn").forEach(btn => {
  btn.addEventListener("click", function () {
    document.querySelectorAll(".speed-btn").forEach(b => b.classList.remove("active"));
    this.classList.add("active");
  });
});

document.querySelectorAll(".preset-btn").forEach(btn => {
  btn.addEventListener("click", function () {
    const [row, col, dist, tour] = this.dataset.preset.split(",").map(Number);
    applyPreset(row, col, dist, tour);
  });
});


window.addEventListener("resize", resizeCanvases);
initEnergyChart();
randomizeCities();
setTimeout(resizeCanvases, 50);
