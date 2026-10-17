import { capySVG, robotSVG, logoSVG } from "./capy.js";

const BASELINE = 472340;
const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const VIDEO = params.has("video");
if (VIDEO) document.body.classList.add("video");

const CAST = [
  { key: "lead", name: "Detective Capy", role: "detective", model: "Nemotron 3.5 Super", job: "lead" },
  { key: "skeptic", name: "Inspector Doubt", role: "skeptic", model: "Qwen 3.8", job: "skeptic" },
  { key: "archivist", name: "Archivist Capy", role: "archivist", model: "Nemotron 3.5 Lightning", job: "runbooks" },
  { key: "field-gpu", name: "Field Capy · GPUs", role: "field", model: "Nemotron 3 Nano", job: "triage" },
  { key: "field-net", name: "Field Capy · network", role: "field", model: "Nemotron 3 Nano", job: "triage" },
  { key: "field-data", name: "Field Capy · data", role: "field", model: "Nemotron 3 Nano", job: "triage" },
  { key: "field-change", name: "Field Capy · changes", role: "field", model: "Nemotron 3 Nano", job: "triage" },
  { key: "governor", name: "Governor", role: "none", model: "rules", job: "loops · budget" },
];
const FLEET = new Set(["nccl_config", "storage_contention", "checkpoint_storage"]);
const rcName = (r) => `${r.type}${r.node && !FLEET.has(r.type) ? " on " + r.node : ""}${r.gpu != null && !FLEET.has(r.type) ? " gpu" + r.gpu : ""}`;
const BASE_CAST = [{ key: "baseline", name: "Regular agent", role: "robot", model: "Nemotron 3.5 Super", job: "single agent, same tools" }];

let S;  // UI state
function reset(agent = "capy") {
  S = { series: [], grid: null, order: [], sus: new Set(), markers: [], toolCount: 0, hyps: {}, t0: null,
        agent, member: {}, result: null, closedAt: null, tAlert: null };
  $("feed").innerHTML = ""; $("board").innerHTML = ""; $("contract").innerHTML = '<li class="muted">waiting for the alert…</li>';
  ["proof", "skeptic", "verify", "alert", "ping", "closed", "score"].forEach((id) => $(id).classList.add("hidden"));
  renderCast(agent === "baseline" ? BASE_CAST : CAST);
  sleepYou(true);
  drawChart(); drawGrid();
}

// ------------------------------------------------------------------ cast
function avatar(m, mood = "idle", yuzu = false) {
  if (m.role === "robot") return robotSVG({ size: 58, mood });
  if (m.key === "governor") return `<svg viewBox="0 0 60 60" width="58" height="58"><circle cx="30" cy="30" r="26" fill="#FEF3C7" stroke="#F59E0B" stroke-width="3"/><path d="M30 14 v17 l10 7" stroke="#B45309" stroke-width="4" fill="none" stroke-linecap="round"/></svg>`;
  return capySVG({ role: m.role, mood, yuzu, size: 58 });
}
function renderCast(list) {
  $("cast").innerHTML = "";
  S.member = {};
  for (const m of list) {
    const el = document.createElement("div");
    el.className = "member";
    el.innerHTML = `<div class="av">${avatar(m)}</div><div class="meta"><div class="who">${m.name}</div>
      <div class="model ${m.model.startsWith("Nemotron") ? "nv" : m.model.startsWith("Qwen") ? "qw" : ""}">${m.model} · ${m.job}</div>
      <div class="bubble muted">zzz…</div></div>`;
    $("cast").appendChild(el);
    S.member[m.key] = { el, m };
  }
  $("modelnote").textContent = list.length > 1 ? "all on Crusoe Managed Inference" : "baseline: one agent";
}
function memberFor(who) { return S.member[who] || S.member[who?.startsWith("field") ? "field-gpu" : "lead"] || Object.values(S.member)[0]; }
function setMood(who, mood, text, yuzu = false) {
  const mm = memberFor(who); if (!mm) return;
  for (const o of Object.values(S.member)) o.el.classList.remove("active");
  mm.el.classList.add("active");
  mm.el.classList.remove("thinking", "alarm", "happy");
  if (mood !== "idle" && mood !== "talk") mm.el.classList.add(mood);
  mm.el.querySelector(".av").innerHTML = avatar(mm.m, mood === "talk" ? "idle" : mood, yuzu);
  const b = mm.el.querySelector(".bubble");
  b.className = "bubble" + (mood === "alarm" ? " alarm" : mood === "happy" ? " happy" : "");
  if (mood === "thinking") { b.innerHTML = `<span class="dots">thinking</span>`; }
  else if (text) b.textContent = text;
}
function sleepYou(asleep) {
  $("you").innerHTML = capySVG({ role: "human", mood: asleep ? "sleep" : "alarm", onsen: true, size: 76 }) + (asleep ? '<span class="zz">z</span>' : "");
  $("you").classList.toggle("awake", !asleep);
}

// ------------------------------------------------------------------ chart
function drawChart() {
  const svg = $("chart"), W = 560, H = 220, P = { l: 44, r: 8, t: 10, b: 22 };
  const pts = S.series;
  if (!pts.length) { svg.innerHTML = `<text x="${W / 2}" y="${H / 2}" text-anchor="middle" fill="#A8A29E">waiting for telemetry…</text>`; return; }
  const tmax = pts[pts.length - 1][0], tmin = Math.max(pts[0][0], tmax - 100 * 60);
  const vis = pts.filter((p) => p[0] >= tmin);
  const ymax = 540000, X = (t) => P.l + (t - tmin) / Math.max(1, tmax - tmin) * (W - P.l - P.r), Y = (v) => H - P.b - v / ymax * (H - P.t - P.b);
  let g = "";
  for (const v of [0, 200000, 400000]) g += `<line x1="${P.l}" x2="${W - P.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="#F1E9DD"/><text x="${P.l - 6}" y="${Y(v) + 4}" text-anchor="end" font-size="10" fill="#A8A29E">${v / 1000}k</text>`;
  g += `<rect x="${P.l}" y="${Y(BASELINE * 1.03)}" width="${W - P.l - P.r}" height="${Y(BASELINE * 0.97) - Y(BASELINE * 1.03)}" fill="#DCFCE7" opacity=".7"/>`;
  g += `<line x1="${P.l}" x2="${W - P.r}" y1="${Y(BASELINE)}" y2="${Y(BASELINE)}" stroke="#A8A29E" stroke-dasharray="5 4"/>`;
  g += `<line x1="${P.l}" x2="${W - P.r}" y1="${Y(BASELINE * 0.97)}" y2="${Y(BASELINE * 0.97)}" stroke="#22C55E" stroke-width="1.2"/>`;
  // smooth: 3-point moving average for display (zero = downtime, kept sharp)
  const sm = vis.map((p, i) => {
    if (p[1] === 0) return p;
    const nb = [vis[i - 1], p, vis[i + 1]].filter((q) => q && q[1] > 0);
    return [p[0], nb.reduce((a, q) => a + q[1], 0) / nb.length];
  });
  g += `<polyline fill="none" stroke="#2563EB" stroke-width="2.2" stroke-linejoin="round" points="${sm.map((p) => `${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(" ")}"/>`;
  for (const m of S.markers) {
    if (m.t < tmin || m.t > tmax + 60) continue;
    const x = Math.min(W - P.r, X(m.t));
    g += `<line x1="${x}" x2="${x}" y1="${P.t}" y2="${H - P.b}" stroke="${m.color}" stroke-dasharray="3 3"/>`;
    g += `<rect x="${Math.min(x + 3, W - 96)}" y="${P.t + m.row * 15}" width="${m.label.length * 5.6 + 8}" height="13" rx="3" fill="${m.color}" opacity=".92"/><text x="${Math.min(x + 7, W - 92)}" y="${P.t + 10 + m.row * 15}" font-size="9.5" fill="#fff" font-weight="700">${m.label}</text>`;
  }
  const lab = (t) => fmtClock(t);
  g += `<text x="${P.l}" y="${H - 6}" font-size="10" fill="#A8A29E">${lab(tmin)}</text><text x="${W - P.r}" y="${H - 6}" font-size="10" fill="#A8A29E" text-anchor="end">${lab(tmax)}</text>`;
  svg.innerHTML = g;
}
function fmtClock(t) {
  const base = Date.UTC(2026, 9, 17, 1, 30, 0);
  return new Date(base + t * 1000).toISOString().substring(11, 16);
}
function addMarker(t, label, color) {
  const row = S.markers.filter((m) => Math.abs(m.t - t) < 600).length % 3;
  S.markers.push({ t, label, color, row });
}

// ------------------------------------------------------------------ grid
function drawGrid() {
  const el = $("gpugrid");
  if (!S.grid) { el.innerHTML = ""; for (let n = 0; n < 66; n++) el.innerHTML += `<div class="node">${'<i class="g"></i>'.repeat(8)}</div>`; return; }
  const codes = S.grid, order = S.order;
  let h = "";
  for (let n = 0; n < order.length; n++) {
    const node = order[n], num = node.split("-").pop();
    const spare = n >= 64;
    let cls = "node" + (spare ? " spare" : "") + (S.sus.has(node) ? " sus" : "");
    let gs = "";
    for (let g = 0; g < 8; g++) {
      const c = codes[n * 8 + g];
      gs += `<i class="g c${c}${S.sus.has(node + "/" + g) ? " sus" : ""}"></i>`;
    }
    h += `<div class="${cls}" title="${node}"><span class="nl">#${num}</span>${gs}</div>`;
  }
  el.innerHTML = h;
}

// ------------------------------------------------------------------ feed + boards
function feed(html, cls = "", clock = "") {
  const d = document.createElement("div");
  d.className = "fi " + cls;
  d.innerHTML = `<span class="t">${clock}</span>${html}`;
  $("feed").prepend(d);
  while ($("feed").children.length > 40) $("feed").lastChild.remove();
}
const esc = (s) => String(s ?? "").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
const NAME = { lead: "Detective", skeptic: "Doubt", archivist: "Archivist", "field-gpu": "Field·GPU", "field-net": "Field·net", "field-data": "Field·data", "field-change": "Field·chg", governor: "Governor", baseline: "Agent", contract: "Detective", capy: "Capy" };
function argStr(a) {
  if (!a || !Object.keys(a).length) return "";
  return Object.entries(a).map(([k, v]) => typeof v === "object" ? `${k}=${JSON.stringify(v).slice(0, 60)}` : `${k}=${v}`).join(", ");
}
function renderContract(items) {
  $("contract").innerHTML = items.map((d) => `<li class="${d.status || "open"}"><span class="ck">${d.status === "pass" ? "✓" : d.status === "fail" ? "✗" : ""}</span>
    <span>${esc(d.check)}${d.note ? `<span class="note">${esc(d.note)}</span>` : ""}</span></li>`).join("");
}
function renderBoard() {
  const order = { confirmed: 0, testing: 1, open: 2 };
  const all = Object.values(S.hyps);
  const live = all.filter((h) => h.status !== "ruled_out").sort((a, b) => (order[a.status] ?? 2) - (order[b.status] ?? 2));
  // ruled-out hypotheses collapse to one line of chips so the proof, skeptic and verify boxes stay on screen
  const out = all.filter((h) => h.status === "ruled_out");
  $("board").innerHTML = live.map((h) => `<div class="hyp ${h.status}"><span class="st">${(h.status || "open").replace("_", " ")}</span>
    ${h.explained_pct > 0 ? `<span class="pct">${Math.round(h.explained_pct)}%</span>` : ""}
    <span class="ht">${esc(h.title)}</span>${h.evidence ? `<div class="ev">${esc(h.evidence)}</div>` : ""}</div>`).join("")
    + (out.length ? `<div class="ruled"><span class="st">ruled out</span>${out.map((h) => `<s title="${esc(h.evidence || "")}">${esc(shortHyp(h.title))}</s>`).join("")}</div>` : "");
}
function shortHyp(t = "") {
  const head = t.split(":")[0].trim();
  return head.length <= 28 ? head : head.slice(0, 26) + "…";
}

// ------------------------------------------------------------------ event handler
function handle(ev, live = false) {
  $("simclock").textContent = ev.clock || $("simclock").textContent;
  switch (ev.type) {
    case "start": reset(ev.agent); break;
    case "tick": {
      if (ev.points?.length) { S.series.push(...ev.points); drawChart(); }
      if (ev.grid) { S.grid = ev.grid.codes; S.order = ev.grid.nodes; drawGrid(); }
      const r = ev.ratio ?? 0;
      $("tps").textContent = `${Math.round(r * BASELINE / 1000)}k tok/s`;
      $("tpsratio").textContent = `${Math.round(r * 100)}% of baseline`;
      $("tpsratio").className = r >= 0.97 ? "good" : "bad";
      const st = $("jobstate"); st.textContent = ev.state; st.className = "state" + (ev.state === "running" ? (r >= .97 ? "" : " bad") : ev.state === "restarting" ? " restart" : " bad");
      break;
    }
    case "alert":
      S.tAlert = ev.t;
      $("alert").className = "alert";
      $("alert").innerHTML = `🚨 ${esc(ev.text)} <span class="route">paged <b>${S.agent === "baseline" ? "the agent" : "Capy"}</b> · you stay asleep</span>`;
      addMarker(ev.t, "page", "#E4572E"); drawChart();
      feed(`<span class="w">PAGE</span>${esc(ev.text)}`, "bad", ev.clock);
      break;
    case "contract":
      renderContract(ev.contract.done_when);
      feed(`<span class="w">Detective</span>wrote the definition of done (${ev.contract.done_when.length} checks)`, "", ev.clock);
      break;
    case "contract_update": renderContract(ev.done_when); break;
    case "thinking": setMood(ev.who, "thinking"); break;
    case "say": setMood(ev.who, ev.mood === "alarm" ? "alarm" : ev.mood === "happy" ? "happy" : "talk", ev.text, ev.mood === "happy" && ev.who !== "skeptic"); break;
    case "tool": {
      S.toolCount++; $("toolcount").textContent = `${S.toolCount} tool calls`;
      for (const h of ev.highlight || []) { if (h.node) S.sus.add(h.node); if (h.node && h.gpu != null) S.sus.add(`${h.node}/${h.gpu}`); }
      if (ev.highlight?.length) drawGrid();
      const extra = ev.explained_pct != null ? ` → <b>${ev.explained_pct}%</b> explained` : "";
      feed(`<span class="w">${NAME[ev.who] || ev.who}</span><code>${ev.name}(${esc(argStr(ev.args))})</code>${extra}`, "", ev.clock);
      if (ev.who !== "skeptic") setMood(ev.who, "thinking");
      break;
    }
    case "triage":
      feed(`<span class="w">Field capys</span>triage sweep done: ${ev.findings.map((f) => f.findings.length).reduce((a, b) => a + b, 0)} findings`, "", ev.clock);
      break;
    case "hypotheses":
      for (const h of ev.items) S.hyps[h.id] = h;
      renderBoard(); break;
    case "proposal": {
      S.proposed = ev.proposal.root_causes || [];
      const rc = (ev.proposal.root_causes || []).map(rcName).join(" + ");
      const plan = (ev.proposal.fix_plan || []).map((a) => a.action).join(", ");
      feed(`<span class="w">Detective</span>proposes <b>${esc(rc)}</b> · fix: <code>${esc(plan || "none")}</code>`, "", ev.clock);
      break;
    }
    case "proof": {
      if (ev.ok && S.proposed?.length) {
        // the proven cause becomes the confirmed card; other open leads are closed out
        for (const h of Object.values(S.hyps)) if (h.status !== "ruled_out" && h.status !== "confirmed") h.status = "ruled_out";
        const conf = Object.values(S.hyps).filter((h) => h.status === "confirmed" && h.id !== "proven");
        if (conf.length) conf.forEach((h) => (h.explained_pct = ev.explained_pct));
        else S.hyps["proven"] = { id: "proven", status: "confirmed", explained_pct: ev.explained_pct,
          title: S.proposed.map((r) => rcName(r).replace(/_/g, " ")).join(" + "),
          evidence: S.proposed.map((r) => r.detail || "").join(" | ").slice(0, 220) };
        renderBoard();
        S.sus = new Set(S.proposed.filter((r) => r.node && !FLEET.has(r.type)).flatMap((r) => [r.node, ...(r.gpu != null ? [`${r.node}/${r.gpu}`] : [])]));
        drawGrid();
      }
      const el = $("proof"); el.classList.remove("hidden");
      el.className = "proof" + (ev.ok ? "" : " fail");
      el.innerHTML = `<span class="big">${Math.round(ev.explained_pct || 0)}%</span><b>${ev.ok ? "Proof gate passed" : "Proof gate: not enough"}</b> — counterfactual from rank timings (needs ≥ 90%)<div class="muted" style="margin-top:4px">${esc(ev.summary)}</div>`;
      feed(`<span class="w">Proof gate</span>${ev.ok ? "✓" : "✗"} ${esc(ev.summary)}`, ev.ok ? "good" : "bad", ev.clock);
      break;
    }
    case "skeptic": {
      const el = $("skeptic"); el.classList.remove("hidden");
      el.className = "skepticbox " + (ev.upheld ? "upheld" : "objection");
      const why = (ev.upheld ? ev.checks : ev.objections) || [];
      el.innerHTML = `<div>${capySVG({ role: "skeptic", mood: ev.upheld ? "happy" : "alarm", size: 52 })}</div><div><div class="verdict">${ev.upheld ? "Upheld" : "Objection!"}</div><div>${esc(why[0] || "")}</div></div>`;
      feed(`<span class="w">Doubt</span>${ev.upheld ? "upheld the diagnosis" : "<b>objected</b>: " + esc((why[0] || "").slice(0, 160))}`, ev.upheld ? "good" : "bad", ev.clock);
      break;
    }
    case "governor":
      feed(`<span class="w">Governor</span><b>${esc(ev.decision)}</b> — ${esc(ev.reason)}`, "gov", ev.clock);
      setMood("governor", ev.decision === "escalate" ? "alarm" : "talk", `${ev.decision}: ${ev.reason}`);
      break;
    case "ping": showPing(ev.request, live); break;
    case "ping_response":
      $("ping").classList.add("hidden"); sleepYou(true);
      feed(`<span class="w">You</span>${ev.approved ? "approved from your phone, went back to sleep" : "denied"}`, ev.approved ? "good" : "bad", ev.clock);
      break;
    case "action": {
      feed(`<span class="w">${S.agent === "baseline" ? "Agent" : "Capy"}</span>executed <code>${ev.name}(${esc(argStr(ev.args))})</code>`, "act", ev.clock);
      const lbl = ev.name === "drain_node" ? `drain #${(ev.args.node || "").split("-").pop()}` : ev.name.replace(/_/g, " ");
      addMarker(ev.t, lbl, ev.disruptive ? "#7C3AED" : "#0EA5E9"); drawChart();
      if (ev.name === "drain_node") S.sus.delete(ev.args.node);
      break;
    }
    case "verify_start": {
      const el = $("verify"); el.classList.remove("hidden", "fail"); el.className = "verifybox";
      el.querySelector(".vlabel").textContent = `Verifying: ${ev.steps} steps at ≥ 97% of baseline…`;
      $("vbar").style.transition = "none"; $("vbar").style.width = "0%";
      requestAnimationFrame(() => { $("vbar").style.transition = `width ${Math.max(0.5, (S.verifyMs || 2500) / 1000)}s linear`; $("vbar").style.width = "92%"; });
      $("vtext").textContent = "";
      break;
    }
    case "verify": {
      const el = $("verify"); el.classList.remove("hidden");
      el.className = "verifybox" + (ev.pass ? "" : " fail");
      $("vbar").style.transition = "width .3s"; $("vbar").style.width = "100%";
      el.querySelector(".vlabel").textContent = ev.pass ? "✓ Verified: the fix held for 200 steps" : "✗ Verification failed — re-planning";
      $("vtext").textContent = ev.summary;
      if (ev.pass) addMarker(ev.t, "verified ✓", "#16A34A"); drawChart();
      feed(`<span class="w">Verifier</span>${ev.pass ? "✓" : "✗"} ${esc(ev.summary)}`, ev.pass ? "good" : "bad", ev.clock);
      break;
    }
    case "ticket": feed(`<span class="w">Capy</span>${esc(ev.text)}`, "", ev.clock); break;
    case "brain": feed(`<span class="w">Archivist</span>added a verified runbook entry to the shared brain`, "good", ev.clock); break;
    case "closed": S.closed = ev.report; showClosed(); break;
    case "escalate":
      $("alert").className = "alert"; $("alert").innerHTML = `⚠️ Escalated to the human: ${esc(ev.reason)}`;
      sleepYou(false); break;
    case "result": S.result = ev.result; if (S.closed) showClosed(); break;
    case "error": feed(`<span class="w">error</span>${esc(ev.text)}`, "bad", ev.clock); break;
  }
}

function showPing(req, live) {
  sleepYou(false);
  const el = $("ping"); el.classList.remove("hidden");
  $("pingcapy").innerHTML = capySVG({ role: "detective", mood: "idle", size: 40 });
  const rc = (req.root_causes || []).map(rcName).join(" + ");
  const plan = (req.plan || []).map((a) => `${a.action}${a.args?.node ? " " + a.args.node : ""}`).join(", ");
  const c = req.cost;
  const line = rc && req.explained_pct != null && c
    ? `Explains ${Math.round(req.explained_pct)}% of the slowdown. Cost: ~${c.downtime_min} min downtime, ${c.rollback_steps} steps rollback (~${c.gpu_hours} GPU-h).`
    : req.text || "";
  $("pingbody").innerHTML = `<b>${esc(rc || req.title)}</b><br>${esc(line)}
    <div class="pc">${req.explained_pct != null ? `<span class="chip">proof: ${Math.round(req.explained_pct)}% explained</span>` : ""}
    ${req.skeptic ? `<span class="chip">skeptic: ${req.skeptic}</span>` : ""}
    ${req.cost ? `<span class="chip cost">~${req.cost.gpu_hours} GPU-h · ${req.cost.downtime_min} min</span>` : ""}</div>
    <div style="margin-top:8px"><b>Plan:</b> ${esc(plan)}</div>`;
  $("approve").onclick = () => answer(true);
  $("deny").onclick = () => answer(false);
  function answer(ok) {
    if (live && LIVE.rid) fetch(`/api/approve/${LIVE.rid}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ approved: ok }) });
  }
  feed(`<span class="w">Capy</span>📱 pinged you once: <b>${esc(plan)}</b>`, "act", "");
}

function showClosed() {
  const r = S.closed || {}, res = S.result || {};
  $("closed").classList.remove("hidden");
  $("closedcapy").innerHTML = S.agent === "baseline" ? robotSVG({ size: 120, mood: "happy" }) : capySVG({ role: "detective", mood: "happy", yuzu: true, size: 130 });
  const metric = (v, l) => v == null ? "" : `<div class="metric"><b>${v}</b><small>${l}</small></div>`;
  $("closedbody").innerHTML = `<h3>${esc(r.title || "Case closed")}</h3><div>${esc(r.summary || "")}</div>
    <div class="kv">${r.root_cause ? `<b>Root cause</b><span>${esc(r.root_cause)}</span>` : ""}${r.proof ? `<b>Proof</b><span>${esc(r.proof)}</span>` : ""}
    ${r.fix ? `<b>Fix</b><span>${esc(r.fix)}</span>` : ""}${r.verification ? `<b>Verified</b><span>${esc(r.verification)}</span>` : ""}</div>
    <div class="metrics">${metric(res.minutes_to_done != null ? res.minutes_to_done + " min" : null, "page → verified done")}
    ${metric(res.pings, res.pings === 1 ? "human ping" : "human pings")}${metric(res.tool_calls, "tool calls")}${metric(res.cost_usd != null ? "$" + res.cost_usd.toFixed(3) : null, "inference on Crusoe")}</div>`;
}

// ------------------------------------------------------------------ replay engine
const DWELL = { tick: 0, thinking: 120, tool: 650, say: 1500, alert: 2600, contract: 2400, contract_update: 250, triage: 900,
  hypotheses: 900, proposal: 1500, proof: 2200, skeptic: 2600, governor: 1500, ping: 3600, ping_response: 1100, action: 1700,
  verify_start: 2600, verify: 2600, ticket: 900, brain: 900, closed: 4200, result: 400, escalate: 2500, start: 300, error: 600 };
const R = { events: [], i: 0, playing: false, speed: 1, timer: null };
function dwell(ev) { let d = DWELL[ev.type] ?? 500; if (ev.type === "say") d += Math.min(2200, (ev.text || "").length * 14); return d; }
function step() {
  if (!R.playing || R.i >= R.events.length) { R.playing = false; $("playbtn").textContent = "▶ Replay"; return; }
  const ev = R.events[R.i++];
  if (ev.type === "verify_start") S.verifyMs = dwell(ev) / R.speed;
  handle(ev);
  if (ev.type === "ping") setTimeout(() => $("approve").classList.add("pressed"), dwell(ev) * 0.6 / R.speed);
  if (ev.type === "ping_response") $("approve").classList.remove("pressed");
  R.timer = setTimeout(step, dwell(ev) / R.speed);
}
async function loadTrace(name) {
  const t = await (await fetch(`/api/trace/${name}`)).json();
  R.events = t.events; R.i = 0;
  reset(t.result?.agent || "capy");
  return t;
}
function play() { if (R.playing) return; R.playing = true; $("playbtn").textContent = "⏸ Pause"; step(); }
function pause() { R.playing = false; clearTimeout(R.timer); $("playbtn").textContent = "▶ Replay"; }

// Deterministic driver for the video: play events [i, end) spread over ms.
function playTo(end, ms) {
  pause();
  return new Promise((resolve) => {
    const evs = R.events.slice(R.i, end);
    const total = evs.reduce((a, e) => a + dwell(e), 0) || 1;
    const k = ms / total;
    let idx = 0;
    const run = () => {
      if (idx >= evs.length) { R.i = end; return resolve(); }
      const ev = evs[idx++]; R.i++;
      if (ev.type === "verify_start") S.verifyMs = dwell(ev) * k * 4;
      handle(ev);
      if (ev.type === "ping") setTimeout(() => $("approve").classList.add("pressed"), dwell(ev) * k * 0.7);
      if (ev.type === "ping_response") $("approve").classList.remove("pressed");
      setTimeout(run, dwell(ev) * k);
    };
    run();
  });
}
function findEvent(type, nth = 0, pred = () => true) {
  let n = 0;
  for (let i = 0; i < R.events.length; i++) if (R.events[i].type === type && pred(R.events[i]) && n++ === nth) return i;
  return -1;
}
function caption(text) { const c = $("caption"); if (!text) return c.classList.add("hidden"); c.textContent = text; c.classList.remove("hidden"); }
async function showScore(split = "test") {
  let s;
  try { s = await (await fetch(`/api/results/${split}`)).json(); } catch { return; }
  const o = s.overall, c = o.capy || {}, b = o.baseline || {};
  const fmt = { pct: (v) => `${v}%`, int: (v) => `${v}`, gpuh: (v) => v.toFixed(0), dec: (v) => v.toFixed(2), usd: (v) => `${(v * 100).toFixed(1)}¢` };
  const row = (label, key, better, f) => {
    const vc = c[key], vb = b[key];
    // differences under 3% are a tie, not a win
    const tie = vc == null || vb == null || Math.abs(vc - vb) <= 0.03 * Math.max(Math.abs(vc), Math.abs(vb));
    const cw = !tie && (better === "low" ? vc < vb : vc > vb), bw = !tie && !cw;
    const show = (v) => (v == null ? "–" : fmt[f](v));
    return `<div class="lab">${label}</div><div class="val ${cw ? "win" : ""}">${show(vc)}</div><div class="val ${bw ? "win" : ""}">${show(vb)}</div>`;
  };
  const skipped = Object.keys(s.excluded || {}).length;
  $("scorebody").innerHTML = `<h2>${c.n} unseen incidents · Capy vs a single agent (same model, same tools)</h2>
    <div class="muted">test split: new seeds, 3 held-out fault types no prompt mentions, and two-fault incidents</div>
    <div class="vs"><div></div><div class="hd">${capySVG({ role: "detective", mood: "happy", yuzu: true, size: 44 })} Capy team</div><div class="hd">${robotSVG({ size: 44 })} Single agent</div>
    ${row("job actually fixed", "fixed_pct", "high", "pct")}
    ${row("false “done” claims", "false_done_count", "low", "int")}
    ${row("wasted disruptive actions", "wasted_disruptive_actions", "low", "int")}
    ${row("correct root cause", "diagnosis_correct_pct", "high", "pct")}
    ${row("GPU-hours lost / incident", "gpu_hours_lost_mean", "low", "gpuh")}
    ${row("human pings / incident", "pings_per_incident", "low", "dec")}
    ${row("inference cost / incident", "cost_usd_mean", "low", "usd")}</div>
    <div class="scorefoot">Simulated 512× H100 cluster with ground-truth faults; agents run live on Crusoe Managed Inference. Green = better by more than 3%.
    ${skipped ? `${skipped} incidents with an unfinished run are listed in results/test/summary.md, not scored.` : ""}</div>`;
  $("score").classList.remove("hidden");
}
function showCard(html) { $("cardbody").innerHTML = html; $("card").classList.remove("hidden"); }
function showTitle(s) {
  const cast = ["detective", "field", "skeptic", "archivist"].map((r) => capySVG({ role: r, mood: r === "detective" ? "happy" : "idle", yuzu: r === "detective", size: 120 })).join("");
  $("titlebody").innerHTML = `<div class="tcast">${cast}</div><h1>${esc(s.title || "Capy")}</h1><p>${esc(s.subtitle || "")}</p>`;
  $("title").classList.remove("hidden");
}
// Video driver: one narration segment = one call, paced to the clip length.
window.capyVideo = async (s, ms) => {
  caption(s.caption ?? s.say);
  if (s.hide) ["closed", "score", "title", "card"].forEach((id) => $(id).classList.add("hidden"));
  if (s.action === "title") showTitle(s);
  else if (s.action === "card") showCard(s.html);
  else if (s.action === "score") showScore(s.split || "test");
  else if (s.action === "play") {
    let end = R.events.length;
    if (s.until) {
      const idx = findEvent(s.until.type, s.until.nth || 0, (e) => !s.until.who || e.who === s.until.who);
      if (idx >= 0) end = idx + 1 + (s.until.after || 0);
    }
    playTo(end, ms * (s.fill ?? 0.9));
  }
};
window.capy = { loadTrace, play, pause, playTo, findEvent, caption, showScore, events: () => R.events, pos: () => R.i,
  hideClosed: () => $("closed").classList.add("hidden"), hideScore: () => $("score").classList.add("hidden") };

// ------------------------------------------------------------------ live mode
const LIVE = { rid: null, es: null };
async function runLive() {
  pause();
  const scen = $("scensel").value;
  const r = await (await fetch("/api/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ scenario: scen, agent: "capy" }) })).json();
  LIVE.rid = r.run_id;
  LIVE.es?.close();
  LIVE.es = new EventSource(`/api/stream/${r.run_id}`);
  LIVE.es.onmessage = (m) => handle(JSON.parse(m.data), true);
  LIVE.es.addEventListener("end", () => LIVE.es.close());
}

// ------------------------------------------------------------------ boot
(async function boot() {
  $("logo").innerHTML = logoSVG(46);
  reset();
  try {
    const traces = await (await fetch("/api/traces")).json();
    $("tracesel").innerHTML = traces.map((t) => `<option>${t}</option>`).join("");
    const scens = await (await fetch("/api/scenarios")).json();
    $("scensel").innerHTML = scens.map((s) => `<option value="${s.id}">${s.id}</option>`).join("");
    const want = params.get("trace") || (traces.includes("demo-capy") ? "demo-capy" : traces[0]);
    if (want) { $("tracesel").value = want; await loadTrace(want); }
  } catch (e) { console.error(e); }
  $("playbtn").onclick = () => (R.playing ? pause() : play());
  $("tracesel").onchange = async (e) => { pause(); await loadTrace(e.target.value); };
  $("speed").onchange = (e) => (R.speed = parseFloat(e.target.value));
  $("livebtn").onclick = runLive;
  if (params.get("speed")) R.speed = parseFloat(params.get("speed"));
  if (params.has("autoplay")) play();
  window.capyReady = true;
})();
