"""
BHU Cross-Match — Interactive Animation for GitHub Codespaces
=============================================================
Generates a fully self-contained HTML file with:
  - Play / Pause
  - Reverse / Forward step buttons
  - Timeline scrubber (drag to any frame)
  - Speed control (0.25× to 4×)
  - Live telemetry (scale factors, deviation, redshift)
  - Side-by-side universe canvases
  - Scale factor + deviation graph

Run:
    python bhu_interactive.py

Output:
    output/bhu_interactive.html

Open in Codespaces:
    1. In the file explorer, right-click bhu_interactive.html
    2. Select "Open with Live Server"   (or install the Live Server extension)
    -- OR --
    1. In the terminal: python -m http.server 8080
    2. Click "Open in Browser" on the Ports tab (port 8080)
    3. Navigate to output/bhu_interactive.html
"""

import os, json
import numpy as np

OUT = "output"
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────────────────
# 1. PHYSICS
# ─────────────────────────────────────────────────────────
z_desi = np.array([0.10, 0.30, 0.51, 0.71, 0.93, 1.32, 2.33])
H_desi = np.array([69.0, 80.5, 90.0, 99.0, 115.5, 150.1, 224.6])

def get_H_emp(z):
    return float(np.interp(z, z_desi, H_desi))

H0, Om, OL, lSE, zQ = 68.0, 0.31, 0.69, 0.085, 1.8

def get_H_theory(z):
    Mz = np.exp(-lSE * z * np.exp(-z / zQ))
    return H0 * np.sqrt(Om * (1 + z)**3 + OL * (1.0 / Mz)**2)

z_start = 2.33
a_start = 1.0 / (1.0 + z_start)
DT      = 0.0001

print("Integrating physics...")
a_A_hist, a_B_hist = [a_start], [a_start]
z_A, z_B = z_start, z_start
a_A, a_B = a_start, a_start

for _ in range(10_000_000):
    a_A += a_A * (get_H_emp(z_A)    / H0) * DT;  z_A = 1.0 / a_A - 1.0
    a_B += a_B * (get_H_theory(z_B) / H0) * DT;  z_B = 1.0 / a_B - 1.0
    a_A_hist.append(a_A)
    a_B_hist.append(a_B)
    if a_A >= 1.0 or a_B >= 1.0:
        break

a_A_arr = np.array(a_A_hist)
a_B_arr = np.array(a_B_hist)
total   = len(a_A_arr)
print(f"Integration complete — {total:,} steps")

# Downsample to 500 keyframes
N_FRAMES = 500
idx      = np.linspace(0, total - 1, N_FRAMES, dtype=int)
aA_kf    = [round(float(a_A_arr[i]), 6) for i in idx]
aB_kf    = [round(float(a_B_arr[i]), 6) for i in idx]

# ─────────────────────────────────────────────────────────
# 2. DETERMINISTIC PARTICLES
# ─────────────────────────────────────────────────────────
rng   = np.random.default_rng(seed=0xBEEF1234)
n_p   = 120
theta = rng.uniform(0, 2 * np.pi, n_p)
r_raw = np.sqrt(rng.uniform(0, 1, n_p))
px    = [round(float(r_raw[i] * np.cos(theta[i])), 5) for i in range(n_p)]
py    = [round(float(r_raw[i] * np.sin(theta[i])), 5) for i in range(n_p)]

# Embed all data as JSON
DATA_JSON = json.dumps({"aA": aA_kf, "aB": aB_kf, "px": px, "py": py})

# ─────────────────────────────────────────────────────────
# 3. HTML TEMPLATE
# ─────────────────────────────────────────────────────────
HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>BHU Cross-Match Interactive</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0d0d0f;color:#ddd;font-family:system-ui,sans-serif;
       display:flex;justify-content:center;padding:1.5rem 1rem}}
  #root{{width:100%;max-width:1000px}}

  h1{{font-size:15px;font-weight:600;color:#eee;letter-spacing:.03em;
      text-align:center;margin-bottom:4px}}
  .sub{{font-size:12px;color:#666;text-align:center;margin-bottom:1.2rem}}

  /* Telemetry */
  .tele{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:1rem}}
  .tel{{background:#15151a;border:1px solid #2a2a35;border-radius:8px;padding:10px 12px}}
  .tel-lbl{{font-size:10px;color:#666;text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px}}
  .tel-val{{font-size:17px;font-weight:600;font-variant-numeric:tabular-nums}}

  /* Universe canvases */
  .univs{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px}}
  .univ-wrap{{position:relative;background:#000;border:1px solid #2a2a35;border-radius:10px;overflow:hidden}}
  .univ-badge{{position:absolute;top:8px;left:8px;font-size:11px;font-weight:500;
               padding:3px 8px;border-radius:5px;z-index:2}}
  .badge-a{{background:rgba(16,185,129,.18);color:#10B981;border:1px solid #10B981}}
  .badge-b{{background:rgba(55,138,221,.18);color:#378ADD;border:1px solid #378ADD}}
  canvas.univ{{display:block;width:100%;height:240px}}

  /* Graph */
  .graph-wrap{{background:#15151a;border:1px solid #2a2a35;border-radius:10px;
               padding:12px;margin-bottom:14px}}
  .graph-legend{{display:flex;gap:18px;font-size:11px;color:#888;margin-bottom:8px;flex-wrap:wrap}}
  .leg{{display:flex;align-items:center;gap:6px}}
  .leg-line{{width:22px;height:2px;border-radius:2px}}
  canvas#graph{{display:block;width:100%;height:180px}}

  /* Controls */
  .controls{{background:#15151a;border:1px solid #2a2a35;border-radius:10px;padding:14px 16px}}

  /* Timeline slider */
  .timeline-row{{display:flex;align-items:center;gap:10px;margin-bottom:12px}}
  .timeline-row label{{font-size:11px;color:#666;white-space:nowrap}}
  #timeline{{flex:1;height:4px;accent-color:#378ADD;cursor:pointer}}
  .frame-lbl{{font-size:11px;color:#888;min-width:60px;text-align:right;
              font-variant-numeric:tabular-nums}}

  /* Buttons + speed */
  .btn-row{{display:flex;gap:8px;align-items:center;flex-wrap:wrap}}
  button{{
    font-size:13px;font-weight:500;padding:6px 14px;
    border-radius:6px;border:1px solid #333;
    background:#1e1e28;color:#ddd;cursor:pointer;
    display:flex;align-items:center;gap:5px;
    transition:background .15s,border-color .15s
  }}
  button:hover{{background:#2a2a38;border-color:#555}}
  button.active{{background:#1a3a5c;border-color:#378ADD;color:#7ec8ff}}
  .sep{{width:1px;height:24px;background:#2a2a35;margin:0 4px}}
  .speed-grp{{display:flex;align-items:center;gap:6px;margin-left:auto}}
  .speed-grp label{{font-size:11px;color:#666}}
  #speed{{width:90px;accent-color:#378ADD}}
  #speed-lbl{{font-size:12px;color:#aaa;min-width:28px}}

  /* Status dot */
  .status{{display:flex;align-items:center;gap:6px;font-size:11px;color:#666;
           margin-top:10px}}
  .dot{{width:7px;height:7px;border-radius:50%;background:#444}}
  .dot.running{{background:#10B981;box-shadow:0 0 6px #10B981}}
  .dot.done{{background:#378ADD}}
</style>
</head>
<body>
<div id="root">
  <h1>BHU Cross-Match · Interactive Simulation</h1>
  <div class="sub">Universe A (Real DESI Data) vs Universe B (BHU Super-Eddington Theory)</div>

  <div class="tele">
    <div class="tel">
      <div class="tel-lbl">Redshift z</div>
      <div class="tel-val" id="tel-z" style="color:#aaa">2.330</div>
    </div>
    <div class="tel">
      <div class="tel-lbl">Scale A (DESI)</div>
      <div class="tel-val" id="tel-a" style="color:#10B981">0.30030</div>
    </div>
    <div class="tel">
      <div class="tel-lbl">Scale B (BHU)</div>
      <div class="tel-val" id="tel-b" style="color:#378ADD">0.30030</div>
    </div>
    <div class="tel">
      <div class="tel-lbl">Deviation</div>
      <div class="tel-val" id="tel-dev" style="color:#BA7517">0.0000%</div>
    </div>
  </div>

  <div class="univs">
    <div class="univ-wrap">
      <div class="univ-badge badge-a">Universe A · DESI data</div>
      <canvas id="canvas-a" class="univ"></canvas>
    </div>
    <div class="univ-wrap">
      <div class="univ-badge badge-b">Universe B · BHU theory</div>
      <canvas id="canvas-b" class="univ"></canvas>
    </div>
  </div>

  <div class="graph-wrap">
    <div class="graph-legend">
      <span class="leg"><span class="leg-line" style="background:#10B981"></span>Universe A (DESI)</span>
      <span class="leg"><span class="leg-line" style="background:#378ADD;background:none;border-top:2px dashed #378ADD"></span>Universe B (BHU)</span>
      <span class="leg"><span class="leg-line" style="background:#E24B4A"></span>Deviation ×10</span>
    </div>
    <canvas id="graph"></canvas>
  </div>

  <div class="controls">
    <!-- Timeline -->
    <div class="timeline-row">
      <label>Timeline</label>
      <input type="range" id="timeline" min="0" max="499" value="0" step="1">
      <span class="frame-lbl" id="frame-lbl">Frame 0 / 499</span>
    </div>

    <!-- Buttons -->
    <div class="btn-row">
      <button id="btn-rew"   title="Step back 10 frames">&#9664;&#9664; Back</button>
      <button id="btn-prev"  title="Step back 1 frame">&#9664; Step</button>
      <button id="btn-play"  class="active">&#9646;&#9646; Pause</button>
      <button id="btn-next"  title="Step forward 1 frame">Step &#9654;</button>
      <button id="btn-fwd"   title="Step forward 10 frames">Fwd &#9654;&#9654;</button>
      <div class="sep"></div>
      <button id="btn-reset">&#8635; Reset</button>

      <div class="speed-grp">
        <label>Speed</label>
        <input type="range" id="speed" min="1" max="8" value="3" step="1">
        <span id="speed-lbl">1×</span>
      </div>
    </div>

    <div class="status">
      <span class="dot running" id="dot"></span>
      <span id="status-txt">Playing</span>
    </div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
// ── Embedded physics data (pre-computed by Python) ──────────
const DATA = {DATA_JSON};
const aA   = DATA.aA;
const aB   = DATA.aB;
const PX   = DATA.px;
const PY   = DATA.py;
const N    = aA.length;   // 500 keyframes

// ── State ────────────────────────────────────────────────────
let frame    = 0;
let playing  = true;
let speedMul = 1.0;
let lastTs   = 0;
const FPS    = 30;
let msPerFrame = 1000 / FPS;
let raf;

// Speed map: slider 1-8 → multiplier
const SPEED_MAP = [0.25, 0.5, 1, 1.5, 2, 3, 4, 6];
const SPEED_LBL = ["0.25×","0.5×","1×","1.5×","2×","3×","4×","6×"];

// ── Chart.js graph ───────────────────────────────────────────
const graphCtx = document.getElementById("graph").getContext("2d");
const BG = "#15151a";
const chart = new Chart(graphCtx, {{
  type: "line",
  data: {{
    labels: aA.map((_,i) => i),
    datasets: [
      {{ label:"A", data: new Array(N).fill(null),
         borderColor:"#10B981", borderWidth:2, pointRadius:0, tension:0.3, yAxisID:"y" }},
      {{ label:"B", data: new Array(N).fill(null),
         borderColor:"#378ADD", borderWidth:2, borderDash:[5,4], pointRadius:0, tension:0.3, yAxisID:"y" }},
      {{ label:"Dev", data: new Array(N).fill(null),
         borderColor:"#E24B4A", borderWidth:1.5, pointRadius:0, tension:0.3, yAxisID:"y2" }}
    ]
  }},
  options: {{
    responsive:true, maintainAspectRatio:false, animation:false,
    plugins:{{ legend:{{display:false}} }},
    scales:{{
      x:{{ display:false }},
      y:{{ min:0, max:1.35, position:"left",
           ticks:{{color:"#666",font:{{size:10}},maxTicksLimit:6}},
           grid:{{color:"rgba(255,255,255,0.04)"}},
           title:{{display:true,text:"Scale factor a(t)",color:"#666",font:{{size:10}}}} }},
      y2:{{ min:-0.5, max:10, position:"right",
            ticks:{{color:"#E24B4A",font:{{size:10}},maxTicksLimit:4}},
            grid:{{display:false}},
            title:{{display:true,text:"Dev% ×10",color:"#E24B4A",font:{{size:10}}}} }}
    }}
  }}
}});

// Pre-fill background full-run data (faint)
chart.data.datasets[0].data = aA.map(v => v);
chart.data.datasets[1].data = aB.map(v => v);
chart.data.datasets[2].data = aA.map((_,i) => +(Math.abs(aA[i]-aB[i])/aA[i]*100*10).toFixed(4));
// Set initial alpha very low — use borderColor opacity trick
chart.data.datasets[0].borderColor = "#10B98122";
chart.data.datasets[1].borderColor = "#378ADD22";
chart.data.datasets[2].borderColor = "#E24B4A22";
chart.update("none");

// Overlay live datasets on top
chart.data.datasets.push(
  {{ label:"A-live", data: new Array(N).fill(null),
     borderColor:"#10B981", borderWidth:2.5, pointRadius:0, tension:0.3, yAxisID:"y" }},
  {{ label:"B-live", data: new Array(N).fill(null),
     borderColor:"#378ADD", borderWidth:2.5, borderDash:[5,4], pointRadius:0, tension:0.3, yAxisID:"y" }},
  {{ label:"Dev-live", data: new Array(N).fill(null),
     borderColor:"#E24B4A", borderWidth:1.8, pointRadius:0, tension:0.3, yAxisID:"y2" }}
);
chart.update("none");
const LA = 3, LB = 4, LD = 5;   // live dataset indices

// Playhead vertical line plugin
const playheadPlugin = {{
  id:"playhead",
  afterDraw(ch) {{
    const x = ch.scales.x.getPixelForValue(frame);
    const {{top, bottom}} = ch.chartArea;
    const ctx = ch.ctx;
    ctx.save();
    ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, bottom);
    ctx.strokeStyle = "rgba(255,255,255,0.35)";
    ctx.lineWidth = 1.5; ctx.setLineDash([4,3]);
    ctx.stroke(); ctx.restore();
  }}
}};
Chart.register(playheadPlugin);

// ── Canvas drawing ───────────────────────────────────────────
function hexRgb(h) {{
  return parseInt(h.slice(1,3),16)+","+parseInt(h.slice(3,5),16)+","+parseInt(h.slice(5,7),16);
}}

function drawUniv(id, af, col) {{
  const cv  = document.getElementById(id);
  const ctx = cv.getContext("2d");
  const W   = cv.width  = cv.offsetWidth;
  const H   = cv.height = cv.offsetHeight;
  const cx=W/2, cy=H/2, mr=Math.min(cx,cy)*0.88;
  ctx.clearRect(0,0,W,H);

  const a_start = aA[0];
  const sc = af;

  // Glow
  const g = ctx.createRadialGradient(cx,cy,0,cx,cy,mr*sc*1.1);
  g.addColorStop(0,"rgba("+hexRgb(col)+",0.08)");
  g.addColorStop(1,"rgba(0,0,0,0)");
  ctx.fillStyle=g; ctx.fillRect(0,0,W,H);

  // Particles
  for (let i=0; i<PX.length; i++) {{
    const r = Math.sqrt(PX[i]*PX[i]+PY[i]*PY[i]);
    const op = Math.floor((0.35+0.65*(1-r))*255).toString(16).padStart(2,"0");
    ctx.beginPath();
    ctx.arc(cx+PX[i]*mr*sc, cy+PY[i]*mr*sc, r<0.3?2.2:1.6, 0, 2*Math.PI);
    ctx.fillStyle = col+op;
    ctx.fill();
  }}

  // Horizon ring
  ctx.beginPath(); ctx.arc(cx,cy,mr*sc,0,2*Math.PI);
  ctx.strokeStyle=col+"44"; ctx.lineWidth=1; ctx.stroke();

  // Redshift label
  const z = (1/af)-1;
  ctx.font="11px system-ui"; ctx.fillStyle=col+"bb";
  ctx.fillText("z = "+z.toFixed(3), 10, H-10);

  // Progress arc (top-right)
  const prog = Math.min((af-a_start)/(1.0-a_start), 1);
  ctx.beginPath();
  ctx.arc(W-18,H-18,9,0,2*Math.PI);
  ctx.strokeStyle=col+"22"; ctx.lineWidth=2.5; ctx.stroke();
  ctx.beginPath();
  ctx.arc(W-18,H-18,9,-Math.PI/2,-Math.PI/2+2*Math.PI*prog);
  ctx.strokeStyle=col+"dd"; ctx.lineWidth=2.5; ctx.stroke();
}}

// ── Render a specific frame ───────────────────────────────────
function renderFrame(f) {{
  f = Math.max(0, Math.min(N-1, Math.round(f)));
  frame = f;

  const vA  = aA[f];
  const vB  = aB[f];
  const dev = Math.abs(vA-vB)/vA*100;
  const z   = (1/vA)-1;

  // Telemetry
  document.getElementById("tel-z").textContent   = z.toFixed(3);
  document.getElementById("tel-a").textContent   = vA.toFixed(5);
  document.getElementById("tel-b").textContent   = vB.toFixed(5);
  const devEl = document.getElementById("tel-dev");
  devEl.textContent = dev.toFixed(4)+"%";
  devEl.style.color = dev<0.5?"#10B981":dev<2?"#BA7517":"#E24B4A";

  // Universe canvases
  drawUniv("canvas-a", vA, "#10B981");
  drawUniv("canvas-b", vB, "#378ADD");

  // Graph live lines
  const liveA = new Array(N).fill(null);
  const liveB = new Array(N).fill(null);
  const liveD = new Array(N).fill(null);
  for (let i=0; i<=f; i++) {{
    liveA[i] = aA[i];
    liveB[i] = aB[i];
    liveD[i] = +(Math.abs(aA[i]-aB[i])/aA[i]*100*10).toFixed(4);
  }}
  chart.data.datasets[LA].data = liveA;
  chart.data.datasets[LB].data = liveB;
  chart.data.datasets[LD].data = liveD;
  chart.update("none");

  // Slider
  const slider = document.getElementById("timeline");
  slider.value = f;
  document.getElementById("frame-lbl").textContent = "Frame "+f+" / "+(N-1);
}}

// ── Animation loop ────────────────────────────────────────────
function tick(ts) {{
  if (!playing) {{ raf=null; return; }}
  const elapsed = ts - lastTs;
  if (elapsed >= msPerFrame) {{
    lastTs = ts;
    if (frame >= N-1) {{
      // reached end — stop
      playing = false;
      setPlayState(false, true);
      raf=null; return;
    }}
    renderFrame(frame + 1);
  }}
  raf = requestAnimationFrame(tick);
}}

function startPlay() {{
  if (!playing) {{
    playing = true;
    lastTs  = 0;
    raf = requestAnimationFrame(tick);
  }}
}}

function pausePlay() {{
  playing = false;
  if (raf) {{ cancelAnimationFrame(raf); raf=null; }}
}}

function setPlayState(isPlaying, done=false) {{
  const btn = document.getElementById("btn-play");
  const dot = document.getElementById("dot");
  const txt = document.getElementById("status-txt");
  if (done) {{
    btn.innerHTML = "&#9654; Play";
    btn.classList.remove("active");
    dot.className="dot done"; txt.textContent="Complete";
  }} else if (isPlaying) {{
    btn.innerHTML = "&#9646;&#9646; Pause";
    btn.classList.add("active");
    dot.className="dot running"; txt.textContent="Playing";
  }} else {{
    btn.innerHTML = "&#9654; Play";
    btn.classList.remove("active");
    dot.className="dot"; txt.textContent="Paused";
  }}
}}

// ── Controls ──────────────────────────────────────────────────
document.getElementById("btn-play").addEventListener("click", () => {{
  if (frame >= N-1 && !playing) {{
    renderFrame(0); startPlay(); setPlayState(true);
  }} else if (playing) {{
    pausePlay(); setPlayState(false);
  }} else {{
    startPlay(); setPlayState(true);
  }}
}});

document.getElementById("btn-prev").addEventListener("click", () => {{
  pausePlay(); setPlayState(false);
  renderFrame(frame - 1);
}});

document.getElementById("btn-next").addEventListener("click", () => {{
  pausePlay(); setPlayState(false);
  renderFrame(frame + 1);
}});

document.getElementById("btn-rew").addEventListener("click", () => {{
  pausePlay(); setPlayState(false);
  renderFrame(frame - 10);
}});

document.getElementById("btn-fwd").addEventListener("click", () => {{
  pausePlay(); setPlayState(false);
  renderFrame(frame + 10);
}});

document.getElementById("btn-reset").addEventListener("click", () => {{
  pausePlay(); setPlayState(false);
  renderFrame(0);
}});

document.getElementById("timeline").addEventListener("input", e => {{
  pausePlay(); setPlayState(false);
  renderFrame(parseInt(e.target.value));
}});

document.getElementById("speed").addEventListener("input", e => {{
  const idx     = parseInt(e.target.value) - 1;
  speedMul      = SPEED_MAP[idx];
  msPerFrame    = 1000 / FPS / speedMul;
  document.getElementById("speed-lbl").textContent = SPEED_LBL[idx];
}});

// Keyboard shortcuts
document.addEventListener("keydown", e => {{
  if (e.code==="Space") {{
    e.preventDefault();
    document.getElementById("btn-play").click();
  }} else if (e.code==="ArrowLeft") {{
    e.preventDefault();
    if (e.shiftKey) document.getElementById("btn-rew").click();
    else            document.getElementById("btn-prev").click();
  }} else if (e.code==="ArrowRight") {{
    e.preventDefault();
    if (e.shiftKey) document.getElementById("btn-fwd").click();
    else            document.getElementById("btn-next").click();
  }}
}});

// ── Init ──────────────────────────────────────────────────────
renderFrame(0);
raf = requestAnimationFrame(tick);
</script>
</body>
</html>"""

# Write output
out_path = os.path.join(OUT, "bhu_interactive.html")
with open(out_path, "w") as f:
    f.write(HTML)

print(f"Saved  →  {out_path}")
print()
print("To open in Codespaces:")
print("  Option 1: Right-click bhu_interactive.html → Open with Live Server")
print("  Option 2: Run  python -m http.server 8080  then open Ports tab → port 8080")
print()
print("Controls:")
print("  Space          Play / Pause")
print("  Arrow Left     Step back 1 frame")
print("  Arrow Right    Step forward 1 frame")
print("  Shift+Left     Jump back 10 frames")
print("  Shift+Right    Jump forward 10 frames")
print("  Timeline bar   Drag to any position")
print("  Speed slider   0.25× to 6×")