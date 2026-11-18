"use strict";
// Drishti control-room UI. Plain JS, no build step: open http://localhost:8000.

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

const EXAMPLES = [
  "a woman in a white t-shirt with red flowers",
  "a man in a black shirt with a patterned vest",
  "a worker in an orange safety vest and yellow helmet",
  "ಬೋಳು ತಲೆಯ ಕನ್ನಡಕ ಹಾಕಿರುವ ಗಂಡಸು",
  "kappu udupu hakiruva udda koodalina hengasu",
  "a woman in a teal jacket and black leggings",
];

const state = {
  cams: {},          // id -> camera (indexed and live)
  links: [],         // site map walkways
  current: null,     // last search response
  liveStops: [],     // live sightings added to the current trail
  watches: new Map(),// watch id -> query
  photo: null,
  lang: "en-IN",
  current_watch: null,
  lastAlert: 0,
};

const fmt = (iso) => iso ? new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }) : "";
const ctx = () => ({ officer: $("#officer").value.trim(), reason: $("#reason").value.trim() });
const caseQS = () => new URLSearchParams(ctx()).toString();

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail ? (typeof data.detail === "string" ? data.detail : "Invalid request") : `Error ${res.status}`);
  return data;
}

// ------------------------------------------------------------------ start-up
async function init() {
  const o = await api("/api/overview");
  o.cameras.forEach((c) => { state.cams[c.id] = c; });
  state.links = o.links;
  $("#nCams").textContent = o.cameras.length;
  $("#nPeople").textContent = o.people_indexed;
  if (!o.privacy_blur) { $("#privPill").textContent = "Privacy blur off"; $("#privPill").classList.remove("priv"); }
  for (const e of EXAMPLES) {
    const c = document.createElement("span");
    c.className = "chip" + (/[ಀ-೿]/.test(e) ? " kn" : "");
    c.textContent = e;
    c.onclick = () => { $("#q").value = e; clearPhoto(); run(); };
    $("#chips").appendChild(c);
  }
  drawMap();
  loadAudit();
  listenForAlerts();
  refreshWatches();
}

// ------------------------------------------------------------------ search
async function run() {
  const q = $("#q").value.trim();
  if (!q && !state.photo) return;
  $("#meta").innerHTML = `<span class="spinner"></span> Searching ${Object.keys(state.cams).length} cameras…`;
  try {
    let data;
    if (state.photo) {
      const fd = new FormData();
      fd.append("file", state.photo); fd.append("query", q);
      fd.append("officer", ctx().officer); fd.append("reason", ctx().reason);
      data = await api("/api/search_image", { method: "POST", body: fd });
    } else {
      data = await api("/api/search", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, ...ctx() }) });
    }
    state.current = data;
    state.liveStops = [];
    state.current_watch = null;
    render();
  } catch (err) {
    $("#meta").textContent = err.message;
  }
  loadAudit();
}

function render() {
  const d = state.current;
  const strong = d.results.filter((r) => r.label === "strong").length;
  let meta = `<span>${d.results.length} candidates · <b style="color:var(--ok)">${strong} strong</b> · ${d.latency_ms} ms</span>`;
  if (d.language !== "en") meta += `<span class="understood">${d.language === "kn" ? "ಕನ್ನಡ" : "Kanglish"} understood as: <b>${esc(d.understood_as)}</b></span>`;
  if (d.attributes.length) meta += `<span>Matching all of:</span>` + d.attributes.map((a) => `<span class="attr">${esc(a.replace(/^a person /, ""))}</span>`).join("");
  if (d.trail.length > 1) meta += `<span>Trail across ${d.trail.length} cameras · same-person check passed</span>`;
  $("#meta").innerHTML = meta;

  $("#grid").innerHTML = "";
  if (!d.results.length) $("#grid").innerHTML = `<div class="empty"><b>No one indexed yet.</b>Run the ingest step first.</div>`;
  d.results.forEach((r, i) => {
    const el = document.createElement("div");
    el.className = "res" + (r.on_trail ? " trail" : "");
    el.innerHTML = `<span class="badge ${r.label}">${r.label}</span><span class="rank">#${i + 1}</span>
      <img src="/api/thumb/${r.thumb}" loading="lazy" alt="Person on ${esc(r.camera)}">
      <div class="info"><div class="cam">${esc(r.camera)}${r.on_trail ? '<span class="trailtag">ON TRAIL</span>' : ""}</div>
      <div class="time">${fmt(r.clock_start)} · ${Math.max(1, Math.round(r.t_end - r.t_start))}s</div></div>`;
    el.onclick = () => openResult(r);
    $("#grid").appendChild(el);
  });
  renderTrail();
}

function trailStops() {
  if (!state.current) return [];
  return [...state.current.trail, ...state.liveStops];
}

function renderTrail() {
  const stops = trailStops();
  drawMap(stops);
  const L = stops[stops.length - 1];
  if (!L) {
    $("#last").innerHTML = state.current ? `<div class="note">No confident sighting yet. Try adding more detail.</div>` : "";
    $("#tl").innerHTML = "";
    return;
  }
  const img = L.live ? `/api/alerts/${L.alert}/snapshot` : `/api/thumb/${L.thumb}`;
  $("#last").innerHTML = `<div class="last${L.live ? " live" : ""}"><img src="${img}" alt=""><div>
     <div class="lbl">LAST SEEN${L.live ? '<span class="livebadge">LIVE</span>' : ""}</div>
     <div class="big">${esc(L.camera)}</div><div>${fmt(L.clock_end)}</div></div></div>`;
  $("#tl").innerHTML = stops.map((t) => `<div class="${t.live ? "live" : ""}">${esc(t.camera)}${t.live ? " · live alert" : ""}<span>${fmt(t.clock_start)}${t.clock_end !== t.clock_start ? " – " + fmt(t.clock_end) : ""}</span></div>`).join("");
}

// ------------------------------------------------------------------ map
function drawMap(stops = trailStops()) {
  const C = state.cams;
  let h = `<defs><pattern id="g" width="5" height="5" patternUnits="userSpaceOnUse"><path d="M5 0H0V5" fill="none" stroke="#1d2734" stroke-width=".2"/></pattern>
    <marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10z" fill="#ffb020"/></marker></defs>
    <rect x="2" y="4" width="96" height="72" rx="4" fill="url(#g)" stroke="#2a3748" stroke-width=".4"/>`;
  for (const [a, b] of state.links) {
    if (C[a] && C[b]) h += `<line x1="${C[a].x}" y1="${C[a].y}" x2="${C[b].x}" y2="${C[b].y}" stroke="#223042" stroke-width="2.2" stroke-linecap="round"/>`;
  }
  for (let i = 1; i < stops.length; i++) {
    const a = C[stops[i - 1].camera_id], b = C[stops[i].camera_id];
    if (!a || !b || a === b) continue;
    h += `<line x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}" stroke="${stops[i].live ? "#ff4d4d" : "#ffb020"}" stroke-width=".9" stroke-dasharray="2 1.2" marker-end="url(#arr)"><animate attributeName="stroke-dashoffset" from="6" to="0" dur="1s" repeatCount="indefinite"/></line>`;
  }
  const hit = new Set(stops.map((t) => t.camera_id));
  const last = stops.length ? stops[stops.length - 1].camera_id : null;
  for (const c of Object.values(C)) {
    const on = hit.has(c.id), isLast = c.id === last;
    const colour = isLast ? "#ff6a3d" : on ? "#ffb020" : c.live ? "#7a3a3a" : "#3a4a5e";
    if (isLast) h += `<circle cx="${c.x}" cy="${c.y}" r="3" fill="none" stroke="#ff6a3d" stroke-width=".6"><animate attributeName="r" from="2.5" to="7" dur="1.4s" repeatCount="indefinite"/><animate attributeName="opacity" from="1" to="0" dur="1.4s" repeatCount="indefinite"/></circle>`;
    h += `<circle cx="${c.x}" cy="${c.y}" r="${on ? 2.4 : 1.6}" fill="${colour}"/>`;
    h += `<text x="${c.x}" y="${c.y - 3.6}" font-size="2.8" text-anchor="middle" fill="${on ? "#e8edf3" : "#7d8a9b"}">${esc(c.name)}${c.live ? " ●" : ""}</text>`;
  }
  h += `<text x="5" y="74" font-size="2.4" fill="#4d5d72">SITE MAP · DEMO VENUE · ● LIVE CAMERA</text>`;
  $("#map").innerHTML = h;
}

// ------------------------------------------------------------------ detail dialog
let open = null;
async function openResult(r) {
  open = r;
  $("#dFrame").src = `/api/frame/${r.thumb}`; $("#dFrame").style.display = "block";
  $("#dCrop").src = `/api/thumb/${r.thumb}`;
  const v = $("#dVideo"); v.pause(); v.style.display = "none";
  $("#dCam").textContent = `${r.camera} (${r.camera_id})`;
  $("#dTime").textContent = `On camera ${fmt(r.clock_start)} – ${fmt(r.clock_end)}`;
  $("#dSimilar").innerHTML = `<span class="spinner"></span>`;
  $("#dlg").showModal();
  try {
    const s = await api(`/api/similar/${encodeURIComponent(r.track)}`);
    $("#dSimilar").innerHTML = s.matches.slice(0, 6).map((m) =>
      `<div class="${m.likely_same ? "same" : ""}"><img src="/api/thumb/${m.thumb}" alt="">${esc(m.camera)}<br>${m.likely_same ? "likely same" : Math.round(m.similarity * 100) + "%"}</div>`).join("");
  } catch { $("#dSimilar").textContent = "Not available."; }
}
$("#dPlay").onclick = () => {
  const v = $("#dVideo");
  v.src = `/api/video/${open.camera_id}?${caseQS()}#t=${Math.max(0, open.t - 2)}`;
  v.style.display = "block"; $("#dFrame").style.display = "none"; v.play();
  setTimeout(loadAudit, 800);
};
$("#dReveal").onclick = () => {
  const q = `reveal=1&${caseQS()}`;
  $("#dCrop").src = `/api/thumb/${open.thumb}?${q}`;
  $("#dFrame").src = `/api/frame/${open.thumb}?${q}`;
  setTimeout(loadAudit, 800);
};
$("#dClose").onclick = () => { $("#dVideo").pause(); $("#dlg").close(); };

// ------------------------------------------------------------------ live watch
async function startWatch() {
  const q = $("#q").value.trim();
  if (!q) { $("#q").focus(); return; }
  try {
    const w = await api("/api/watch", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: q, ...ctx() }) });
    state.watches.set(w.id, w.query);
    if (state.current && state.current.query === q) state.current_watch = w.id;
    showLive(true);
    refreshWatches();
    loadAudit();
  } catch (err) { $("#watchList").textContent = err.message; }
}

async function refreshWatches() {
  const r = await api("/api/watches");
  r.watches.forEach((w) => state.watches.set(w.id, w.query));
  $("#watchList").innerHTML = r.watches.length
    ? r.watches.map((w) => `<div class="watchitem"><span>🔔 ${esc(w.query)}</span><button title="Stop watching" data-id="${w.id}">×</button></div>`).join("")
    : "No active watches. Search for someone, then press 🔔 Watch live.";
  $("#watchList").className = r.watches.length ? "" : "note";
  document.querySelectorAll(".watchitem button").forEach((b) => {
    b.onclick = async () => { await fetch(`/api/watch/${b.dataset.id}`, { method: "DELETE" }); state.watches.delete(+b.dataset.id); refreshWatches(); };
  });
}

let liveOn = false;
function showLive(on) {
  liveOn = on;
  $("#liveToggle").textContent = on ? "Hide live feed" : "Show live feed";
  $("#liveFeeds").innerHTML = on ? Object.values(state.cams).filter((c) => c.live).map((c) =>
    `<div class="livefeed"><img src="/api/live/${c.id}.mjpg" alt="Live feed from ${esc(c.name)}"><span><span class="livebadge">LIVE</span> ${esc(c.name)}</span></div>`).join("") : "";
}
$("#liveToggle").onclick = () => showLive(!liveOn);

function listenForAlerts() {
  const es = new EventSource(`/api/events?after=${state.lastAlert}`);
  es.addEventListener("alert", (e) => {
    const a = JSON.parse(e.data);
    state.lastAlert = Math.max(state.lastAlert, a.id);
    onAlert(a);
  });
}

function onAlert(a) {
  const item = document.createElement("div");
  item.className = "alert";
  item.innerHTML = `<img src="/api/alerts/${a.id}/snapshot" alt=""><div><b>MATCH · ${esc(a.camera)}</b><br>${esc(a.query)}<br><span class="note">${fmt(a.time)} · matched on ${esc(a.signal)}</span></div>`;
  $("#alerts").prepend(item);
  const t = $("#toast");
  t.innerHTML = `<img src="/api/alerts/${a.id}/snapshot" alt=""><div><div class="t1">🔔 LIVE MATCH</div><div class="t2">${esc(a.camera)}</div><div>${esc(a.query)}</div><div class="note">${fmt(a.time)} · ${esc(a.signal)}</div></div>`;
  t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 8000);
  // extend the current case's trail when the alert is for the person on screen
  const forCurrent = state.current && (a.watch_id === state.current_watch || a.query === state.current.query);
  if (forCurrent) {
    state.liveStops.push({ camera_id: a.camera_id, camera: a.camera, clock_start: a.time, clock_end: a.time, live: true, alert: a.id });
    renderTrail();
  }
}

// ------------------------------------------------------------------ audit
async function loadAudit() {
  const r = await api("/api/audit?limit=30");
  $("#audit").innerHTML = r.entries.length ? r.entries.map((e) =>
    `<div><b>${esc(e.action.replace("_", " "))}</b> · ${esc(e.officer)} · ${esc(e.ts.slice(11))}<br>${e.query ? "“" + esc(e.query) + "”" : esc(e.camera || e.crop || "")} <i>${esc(e.reason)}</i></div>`).join("") : "No activity yet.";
}

// ------------------------------------------------------------------ inputs
$("#go").onclick = run;
$("#watch").onclick = startWatch;
$("#q").addEventListener("keydown", (e) => { if (e.key === "Enter") run(); });

function clearPhoto() { state.photo = null; $("#photo").value = ""; $("#photoPreview").style.display = "none"; }
$("#photo").onchange = (e) => {
  state.photo = e.target.files[0] || null;
  if (state.photo) { $("#photoPreview").src = URL.createObjectURL(state.photo); $("#photoPreview").style.display = "block"; run(); }
};
$("#photoPreview").onclick = clearPhoto;

// Voice input (Chrome / Edge). English (India) or Kannada.
$("#lang").onclick = () => {
  state.lang = state.lang === "en-IN" ? "kn-IN" : "en-IN";
  $("#lang").textContent = state.lang === "en-IN" ? "EN" : "ಕನ್ನಡ";
  $("#lang").classList.toggle("kn", state.lang === "kn-IN");
};
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
if (SR) {
  const rec = new SR();
  rec.interimResults = true;
  rec.onresult = (e) => { $("#q").value = [...e.results].map((r) => r[0].transcript).join(""); };
  rec.onend = () => { $("#mic").classList.remove("listening"); if ($("#q").value.trim()) run(); };
  $("#mic").onclick = () => { rec.lang = state.lang; $("#mic").classList.add("listening"); rec.start(); };
} else {
  $("#mic").style.display = "none";
  $("#lang").style.display = "none";
}

init();
