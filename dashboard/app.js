const POLL_MS = 1000;
const seen = new Set();
const $ = (id) => document.getElementById(id);
const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "--" : Number(v).toFixed(d));
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const sevClass = (s) => ({ HIGH: "sev-high", ELEVATED: "sev-elevated", WATCH: "sev-watch" }[s] || "sev-nominal");
const SHORT = { Accelerometer1RMS: "accel 1", Accelerometer2RMS: "accel 2", Current: "current", Pressure: "pressure", Temperature: "motor temp", Thermocouple: "fluid temp", Voltage: "voltage", "Volume Flow RateRMS": "flow rate" };

function phaseLabel(s) {
  if (s.phase === "learning") return `Learning baseline (${Math.min(300, s.tick)}/300 rows)`;
  if (s.phase === "calibrating") return "Calibrating alarm threshold";
  return s.severity === "NOMINAL" ? "NOMINAL" : s.severity;
}

function sitePanel(s) {
  const vis = s.vision || { available: false };
  const cls = s.warmup ? "sev-nominal" : sevClass(s.severity);
  const tele = Object.entries(s.telemetry || {}).map(([k, v]) => `<div><span>${SHORT[k] || k}</span>${fmt(v, 3)}</div>`).join("");
  const chain = (s.ledger_tail || []).slice().reverse().map((e, i) => {
    const broken = s.ledger_break_index !== null && s.ledger_break_index !== undefined && e.seq >= s.ledger_break_index;
    return `<div class="chain-link ${i === 0 ? "newest" : ""} ${broken ? "broken" : ""}"><span>#${e.seq}</span><span class="hash">${e.hash}</span><span style="color:var(--text-dim)">${esc(e.type || "")}</span></div>`;
  }).join("");
  const triage = s.triage ? `<span class="chip ${s.triage === "sensor_fault_suspected" ? "chip-sensor" : "chip-process"}">${s.triage === "sensor_fault_suspected" ? "possible sensor fault" : "process anomaly"} &middot; top-channel share ${fmt(s.share, 2)}</span>` : "";
  const label = s.dataset_label_anomaly ? `<span class="chip chip-label">dataset label: anomaly</span>` : "";
  return `
  <div class="site-panel ${cls}">
    <div class="site-head"><span class="site-tag">${s.site_id}</span><div class="site-lamp"><span class="lamp-dot"></span>${phaseLabel(s)}</div></div>
    <div class="replay-line">replaying real pump recording <b>${esc(s.data_file)}</b> &middot; row ${s.tick}/${s.rows_total} &middot; pass ${s.cycle + 1}</div>
    <div class="telemetry-grid">${tele}</div>
    <div class="gauge-row"><div class="gauge-label"><span>Error vs alarm threshold</span><span>${s.warmup ? "--" : fmt(s.ratio, 1) + "x"}</span></div>
      <div class="gauge-track"><div class="gauge-fill score" style="width:${Math.min(100, (s.score || 0) * 100)}%"></div></div></div>
    <div class="chips">${triage}${label}</div>
    <div class="vision-row"><span class="v-title">Telemetry model: ${s.model_params || "--"}-parameter autoencoder &middot; ${esc(s.backend)} &middot; ${s.telemetry_latency_us ? fmt(s.telemetry_latency_us, 0) + " &micro;s/row" : "not trained yet"}</span>
      <span class="v-title">Camera model: YOLOX-nano &middot; ${vis.available ? esc(vis.provider) : "model file missing"}</span>
      ${vis.available && vis.persons !== null ? `persons <b>${vis.persons}</b> &nbsp; vehicles <b>${vis.vehicles}</b> &nbsp; ${fmt(vis.latency_ms, 0)} ms` : "awaiting first frame&hellip;"}</div>
    <div class="note-box ${s.note ? "active" : ""}">${s.note ? esc(s.note) : (s.warmup ? "Learning this site's own normal behaviour before scoring begins." : "Nominal. No active alarm.")}</div>
    <div class="ledger"><div class="ledger-label"><span>Tamper-evident log &middot; ${s.ledger_len} entries</span>
      <span class="ledger-status ${s.ledger_intact ? "ok" : "broken"}">${s.ledger_intact ? "CHAIN INTACT" : "TAMPER DETECTED @#" + s.ledger_break_index}</span></div><div class="chain">${chain}</div></div>
    <button class="tamper-btn" onclick="post('/api/tamper/${s.site_id}')">Simulate tamper on this ledger</button>
  </div>`;
}

async function post(path) { await fetch(path, { method: "POST" }); }
window.post = post;

function feedLog(state) {
  const el = $("event-log");
  const add = (kind, ts, site, msg) => {
    const key = `${kind}|${site}|${ts}|${msg}`; if (seen.has(key)) return; seen.add(key);
    const d = document.createElement("div"); d.className = `event-row ${kind}`;
    d.innerHTML = `<span class="ts">${ts}</span><span class="site">${esc(site)}</span><span class="msg">${esc(msg)}</span>`; el.appendChild(d);
  };
  for (const s of Object.values(state.sites || {})) if (s.note) add("anomaly", `row ${s.tick - (s.tick % 40)}`, s.site_id, s.note);
  for (const e of (state.swarm && state.swarm.sync_log) || []) add(e.kind === "alert" ? "alert" : "sync", new Date(e.ts * 1000).toLocaleTimeString(), "SWARM", e.note);
  while (el.children.length > 200) el.removeChild(el.firstChild);
}

async function poll() {
  try {
    const state = await (await fetch("/api/state")).json();
    const sites = Object.values(state.sites || {});
    if (sites.length) {
      const v = sites[0].vision || {};
      $("provider-line").innerHTML = `Offline-first fleet intelligence &middot; vision provider: ${v.available ? esc(v.provider) : "n/a"} &middot; telemetry provider: ${esc(sites[0].backend)}`;
      $("site-row").innerHTML = sites.map(sitePanel).join("");
      const ids = sites.map((s) => s.site_id);
      $("fed-controls").innerHTML = `
        <button class="tamper-btn" onclick="post('/api/fed/run')">Run federated round</button>
        <button class="tamper-btn" onclick="post('/api/fed/poison/${ids[0]}')">Round with poisoned update from ${ids[0]}</button>
        <button class="tamper-btn" onclick="post('/api/fed/tamper/${ids[1]}')">Round with ${ids[1]} update altered in transit</button>`;
    }
    $("model-version").textContent = (state.swarm && state.swarm.model_version) || "v0";
    $("round-number").textContent = (state.swarm && state.swarm.round_number) || 0;
    $("sites-online").textContent = `${sites.filter((s) => !s.warmup).length}/${sites.length}`;
    feedLog(state);
  } catch (e) { /* server restarting */ } finally { setTimeout(poll, POLL_MS); }
}
poll();
