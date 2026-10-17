// Draws docs/architecture.svg (the README diagram) in the console's style.
//   node scripts/architecture_svg.mjs
import fs from "node:fs";
import path from "node:path";
import { capySVG } from "../web/capy.js";

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const W = 1200, H = 690;
const C = { bg: "#FBF6EE", card: "#FFFFFF", ink: "#2B2118", muted: "#8A7B6C", line: "#EADFD0", capy: "#9C6B44",
  nv: "#76B900", nvBg: "#F1F8E2", nvInk: "#3F6212", qw: "#7C3AED", qwBg: "#F3E8FF", qwInk: "#5B21B6",
  ok: "#7BC47F", gate: "#F0FDF4", gateLine: "#86EFAC", gateInk: "#15803D", amber: "#F5B841", amberBg: "#FFF7E6", amberInk: "#9A3412" };
const FONT = `Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif`;

const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const text = (x, y, s, { size = 13, weight = 400, fill = C.ink, anchor = "start", extra = "" } = {}) =>
  `<text x="${x}" y="${y}" font-size="${size}" font-weight="${weight}" fill="${fill}" text-anchor="${anchor}" ${extra}>${esc(s)}</text>`;
const caps = (x, y, s, fill = C.capy) => text(x, y, s.toUpperCase(), { size: 11, weight: 700, fill, extra: `letter-spacing="1.2"` });
const card = (x, y, w, h, { fill = C.card, stroke = C.line, rx = 16 } = {}) =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" stroke="${stroke}" stroke-width="1.5"/>`;
const capy = (x, y, size, opts) => capySVG({ size, ...opts }).replace(`<svg class="capy-svg"`, `<svg x="${x}" y="${y}"`).replace(` aria-hidden="true"`, "");
const textW = (s, size, weight = 400) => s.length * size * (weight >= 700 ? 0.62 : 0.56);
function chip(xEnd, y, label, kind) {
  const [bg, ink] = { nv: [C.nvBg, C.nvInk], qw: [C.qwBg, C.qwInk], gate: ["#DCFCE7", C.gateInk], amber: [C.amberBg, C.amberInk] }[kind];
  const w = Math.round(textW(label, 11, 700) + 18);
  return `<rect x="${xEnd - w}" y="${y}" width="${w}" height="20" rx="10" fill="${bg}"/>` +
    text(xEnd - w / 2, y + 14, label, { size: 11, weight: 700, fill: ink, anchor: "middle" });
}
function arrow(x1, y1, x2, y2, { color = C.capy, dash = false, both = false, label = "", ly = -8, w = 2 } = {}) {
  const id = both ? "ab" : "a";
  return `<path d="M${x1} ${y1} L${x2} ${y2}" stroke="${color}" stroke-width="${w}" fill="none" ${dash ? `stroke-dasharray="5 4"` : ""}
    marker-end="url(#${id}-${color.slice(1)})" ${both ? `marker-start="url(#${id}s-${color.slice(1)})"` : ""}/>` +
    (label ? text((x1 + x2) / 2, Math.min(y1, y2) + ly, label, { size: 11, weight: 600, fill: color, anchor: "middle" }) : "");
}
const markers = [C.capy, C.muted, C.amberInk, C.nvInk].map((c) => `
  <marker id="a-${c.slice(1)}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="${c}"/></marker>
  <marker id="ab-${c.slice(1)}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="${c}"/></marker>
  <marker id="abs-${c.slice(1)}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="${c}"/></marker>`).join("");

const out = [];
out.push(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" font-family="${FONT}">
<title>Capy architecture: the training job on Crusoe, the Capy on-call loop, and Crusoe Managed Inference</title>
<defs>${markers}</defs>
<rect width="${W}" height="${H}" rx="24" fill="${C.bg}"/>`);

// ---------------------------------------------------------------- left: the job on Crusoe
const L = { x: 20, y: 20, w: 290, h: 610 };
out.push(card(L.x, L.y, L.w, L.h));
out.push(caps(40, 48, "Crusoe Cloud"));
out.push(text(40, 74, "Your training job", { size: 18, weight: 800 }));
out.push(text(40, 95, "512× H100 · Crusoe Managed Kubernetes", { size: 12.5, fill: C.muted }));
out.push(`<rect x="40" y="105" width="138" height="20" rx="10" fill="${C.amberBg}"/>` +
  text(109, 119, "simulated in this repo", { size: 11, weight: 700, fill: C.amberInk, anchor: "middle" }));
// GPU grid: one slow GPU paces 511 others
for (let r = 0; r < 4; r++) for (let c = 0; c < 16; c++) {
  const slow = r === 1 && c === 10;
  out.push(`<rect x="${40 + c * 15}" y="${140 + r * 15}" width="12" height="12" rx="3" fill="${slow ? C.amber : C.ok}" ${slow ? `stroke="#B45309" stroke-width="1.5"` : `opacity=".85"`}/>`);
}
out.push(text(40, 218, "one slow GPU paces all the others", { size: 11.5, fill: C.muted, extra: `font-style="italic"` }));
out.push(caps(40, 252, "Telemetry", C.muted));
["DCGM GPU metrics", "InfiniBand counters", "JournalD · NCCL · trainer logs", "per-rank flight recorder", "change log · audit log", "Instance Health"]
  .forEach((s, i) => out.push(text(40, 275 + i * 21, "· " + s, { size: 13 })));
out.push(caps(40, 418, "Remediation", C.muted));
["drain node → hot spare", "restart (exclude node / config)", "revert a rollout", "reset GPU power limit", "enable local NVMe cache", "support ticket + diagnostics"]
  .forEach((s, i) => out.push(text(40, 441 + i * 21, "· " + s, { size: 13 })));
out.push(text(40, 590, "Crusoe heals hard failures (Instance", { size: 11.5, fill: C.muted }));
out.push(text(40, 606, "Health, AutoClusters). Capy finds the slow ones.", { size: 11.5, fill: C.muted }));

// ---------------------------------------------------------------- middle: the loop
const M = { x: 360, y: 20, w: 500, h: 610 };
out.push(card(M.x, M.y, M.w, M.h));
out.push(caps(380, 48, "Capy · the on-call loop"));
out.push(text(380, 74, "Done means done", { size: 18, weight: 800 }));
out.push(text(380, 95, "LLM steps propose; deterministic gates decide", { size: 12.5, fill: C.muted }));
out.push(capy(782, 26, 64, { role: "detective", mood: "happy", yuzu: true }));
const ROW0 = 118, PITCH = 58, RX = 406, RW = 434;
const rows = [
  ["Contract", "definition of done, written before investigating", "Nemotron 3.5 Super", "nv"],
  ["Triage", "4 field capys sweep GPUs, network, data, changes", "Nemotron 3 Nano ×4", "nv"],
  ["Investigate", "hypotheses tested with read-only tools", "Nemotron 3.5 Super", "nv"],
  ["Proof gate", "counterfactual on per-rank timings explains ≥ 90%", "deterministic", "gate"],
  ["Skeptic", "another model family tries to falsify it", "Qwen 3.8", "qw"],
  ["Fix", "safe fixes run; disruptive ones need 1 approval", "≤ 1 ping", "amber"],
  ["Verify", "200 measured steps at ≥ 97% of baseline", "deterministic", "gate"],
  ["Close", "case file + runbook for the shared brain", "Nemotron 3.5 Lightning", "nv"],
];
const rowY = (i) => ROW0 + i * PITCH, rowMid = (i) => rowY(i) + 25;
out.push(`<path d="M${RX + 22} ${rowMid(0)} V${rowMid(7)}" stroke="${C.capy}" stroke-width="2" opacity=".5"/>`);
rows.forEach(([t, d, ch, kind], i) => {
  const y = rowY(i), gate = kind === "gate";
  out.push(card(RX, y, RW, 50, { fill: gate ? C.gate : C.card, stroke: gate ? C.gateLine : C.line, rx: 12 }));
  out.push(`<circle cx="${RX + 22}" cy="${y + 25}" r="13" fill="${gate ? C.gateInk : C.capy}"/>` +
    text(RX + 22, y + 30, String(i + 1), { size: 13, weight: 800, fill: "#fff", anchor: "middle" }));
  out.push(text(RX + 46, y + 21, t, { size: 14.5, weight: 700 }));
  out.push(text(RX + 46, y + 39, d, { size: 12, fill: C.muted }));
  out.push(chip(RX + RW - 10, y + 8, ch, kind));
});
// re-investigate lane: proof < 90%, skeptic objection, failed verification all go back to step 3
const lane = RX - 22;
out.push(`<path d="M${RX} ${rowMid(6)} H${lane + 8} Q${lane} ${rowMid(6)} ${lane} ${rowMid(6) - 8} V${rowMid(2) + 8} Q${lane} ${rowMid(2)} ${lane + 8} ${rowMid(2)} H${RX - 2}"
  stroke="${C.muted}" stroke-width="1.8" fill="none" stroke-dasharray="5 4" marker-end="url(#a-${C.muted.slice(1)})"/>`);
for (const i of [3, 4]) out.push(`<path d="M${RX} ${rowMid(i)} H${lane}" stroke="${C.muted}" stroke-width="1.8" stroke-dasharray="5 4"/>`);
out.push(text(lane - 6, (rowMid(2) + rowMid(6)) / 2, "not proven · objection · failed → back to 3", {
  size: 10.5, weight: 600, fill: C.muted, anchor: "middle", extra: `transform="rotate(-90 ${lane - 6} ${(rowMid(2) + rowMid(6)) / 2})"` }));
out.push(text(380, 604, "A rules-based governor watches every step for loops, stalls and", { size: 11.5, fill: C.muted }));
out.push(text(380, 620, "budget overruns, and escalates to you instead of guessing.", { size: 11.5, fill: C.muted }));

// arrows between the job and Capy
out.push(arrow(L.x + L.w + 2, 330, M.x - 4, 330, { label: "tools" }));
out.push(arrow(M.x - 2, 497, L.x + L.w + 4, 497, { label: "actions", color: C.amberInk }));

// ---------------------------------------------------------------- right: inference + the humans' side
const R = { x: 910, w: 270 };
out.push(card(R.x, 20, R.w, 262));
out.push(caps(930, 48, "Crusoe Cloud"));
out.push(text(930, 74, "Managed Inference", { size: 18, weight: 800 }));
out.push(text(930, 95, "OpenAI-compatible · tools · JSON mode", { size: 12.5, fill: C.muted }));
[["Nemotron 3.5 Super", "lead investigator · contract", C.nv], ["Nemotron 3 Nano", "4 field capys, in parallel", C.nv],
 ["Nemotron 3.5 Lightning", "archivist · runbooks", C.nv], ["Qwen 3.8", "skeptic: a different model family", C.qw]]
  .forEach(([n, r, col], i) => {
    const y = 108 + i * 36;
    out.push(`<circle cx="937" cy="${y + 11}" r="6" fill="${col}"/>` + text(951, y + 15, n, { size: 13, weight: 700 }) +
      text(951, y + 30, r, { size: 12, fill: C.muted }));
  });
out.push(text(930, 266, "prefix cache served 55% of prompt tokens", { size: 11.5, fill: C.nvInk, weight: 600 }));
out.push(arrow(M.x + M.w + 4, 160, R.x - 4, 160, { both: true, label: "API", color: C.nvInk, ly: -10 }));

function side(y, h, title, sub, art) {
  out.push(card(R.x, y, R.w, h, { rx: 14 }));
  out.push(art);
  out.push(text(R.x + 66, y + 22, title, { size: 14, weight: 700 }));
  out.push(text(R.x + 66, y + 40, sub, { size: 12, fill: C.muted }));
}
side(rowY(3), 50, "Web console", "live over SSE · replay · scoreboard",
  `<rect x="${R.x + 14}" y="${rowY(3) + 11}" width="40" height="28" rx="5" fill="#fff" stroke="${C.capy}" stroke-width="2"/>
   <path d="M${R.x + 20} ${rowY(3) + 31} l7 -7 l6 4 l9 -10" stroke="#2563EB" stroke-width="2" fill="none"/>`);
out.push(arrow(RX + RW + 2, rowMid(3), R.x - 4, rowMid(3), { dash: true, color: C.muted, label: "events" }));
side(rowY(5), 50, "You", "asleep · pinged at most once", capy(R.x + 8, rowY(5) - 2, 52, { role: "human", mood: "sleep", onsen: true }));
out.push(arrow(RX + RW + 2, rowMid(5), R.x - 4, rowMid(5), { color: C.amberInk, label: "ping" }));
side(rowY(7), 50, "Shared brain", "runbooks from verified fixes", capy(R.x + 8, rowY(7) - 2, 52, { role: "archivist", mood: "idle" }));
out.push(arrow(RX + RW + 2, rowMid(7), R.x - 4, rowMid(7), { label: "learn" }));

// ---------------------------------------------------------------- legend
const ly = 662;
let lx = 40;
const legend = (art, label) => { out.push(art(lx) + text(lx + 26, ly + 4, label, { size: 12, fill: C.muted })); lx += 26 + textW(label, 12) + 26; };
legend((x) => `<rect x="${x}" y="${ly - 8}" width="18" height="14" rx="4" fill="${C.card}" stroke="${C.line}" stroke-width="1.5"/>`, "LLM step");
legend((x) => `<rect x="${x}" y="${ly - 8}" width="18" height="14" rx="4" fill="${C.gate}" stroke="${C.gateLine}" stroke-width="1.5"/>`, "deterministic gate (code, not a model)");
legend((x) => `<circle cx="${x + 9}" cy="${ly - 1}" r="6" fill="${C.nv}"/>`, "NVIDIA Nemotron");
legend((x) => `<circle cx="${x + 9}" cy="${ly - 1}" r="6" fill="${C.qw}"/>`, "Qwen (skeptic)");
legend((x) => `<path d="M${x} ${ly - 1} h18" stroke="${C.muted}" stroke-width="1.8" stroke-dasharray="5 4"/>`, "re-investigate");
out.push(text(W - 40, ly + 4, "every agent runs on Crusoe Managed Inference", { size: 12, weight: 600, fill: C.capy, anchor: "end" }));

out.push(`</svg>\n`);
fs.writeFileSync(path.join(ROOT, "docs", "architecture.svg"), out.join("\n"));
console.log("wrote docs/architecture.svg");
