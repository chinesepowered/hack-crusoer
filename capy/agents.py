"""Capy: an on-call agent team that babysits GPU training runs, plus a baseline.

Capy's loop is "done means done":

  alert -> contract (definition of done, written first)
        -> triage (parallel field capys)
        -> investigate (lead, with a governor watching for loops/stalls/budget)
        -> proof gate (counterfactual: causes must explain >= 90%)
        -> skeptic (different model family tries to falsify)
        -> fix (one human ping, only if disruptive)
        -> verify (deterministic: >= 97% throughput for 200 steps, no new errors)
        -> close (case file + runbook entry), or re-plan / escalate

The baseline is a single agent with the same model and the same tools.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from collections import defaultdict

from . import sim
from . import tools as T
from .brain import Brain
from .llm import DISPLAY, LLM, MODELS, LLMError
from .sim import (BASELINE_TPS, GPUS_PER_NODE, HEALTHY_STEP_S, N_GPUS, RESTART_S, SWAP_NODE_S,
                  Cluster, clock_str)

POLICY = {"min_ratio": 0.97, "verify_steps": 200, "min_explained": 90.0, "max_pings": 1,
          "max_minutes": 40, "max_tool_calls": 30}
APPROVAL_S = 180.0           # simulated human response time to a ping
LLM_CALL_S = 4.0             # simulated time charged per model call
HORIZON_S = 75 * 60          # incidents are scored at alert + 75 min
HARDWARE_TYPES = {"gpu_thermal", "gpu_hardware_failure", "ib_link", "pcie_link"}
FLEET_TYPES = {"nccl_config", "storage_contention", "checkpoint_storage"}   # no single node to blame

CAST = {
    "lead": ("Detective Capy", "lead"),
    "contract": ("Detective Capy", "contract"),
    "field-gpu": ("Field Capy (GPUs)", "field"),
    "field-net": ("Field Capy (network)", "field"),
    "field-data": ("Field Capy (data)", "field"),
    "field-change": ("Field Capy (changes)", "field"),
    "skeptic": ("Inspector Doubt", "skeptic"),
    "archivist": ("Archivist Capy", "archivist"),
    "governor": ("Governor", None),
    "baseline": ("Regular agent", "baseline"),
}


async def auto_approve(request: dict, run: "Run") -> bool:
    """Benchmark human: reads the ping and approves after APPROVAL_S."""
    run.cluster.advance(APPROVAL_S)
    return True


def alert_text(c: Cluster) -> str:
    t = c.t
    if c.job_state(t) != "running":
        last = c.steps[-1] if c.steps else None
        return (f"[PAGE] {sim.JOB}: no training progress for {int(t - (last[0] if last else t)) // 60} min "
                f"(last step {last[1] if last else '?'} at {clock_str(last[0]) if last else '?'}Z). Cluster {sim.CLUSTER}, 512x H100.")
    tp = c.throughput(t - 300, t) or 0
    return (f"[PAGE] {sim.JOB}: throughput {tp / 1000:.0f}k tok/s is {100 * tp / BASELINE_TPS:.0f}% of the 7-day p50 "
            f"({BASELINE_TPS / 1000:.0f}k) for 5 min. Cluster {sim.CLUSTER}, 512x H100, us-east1-a.")


def _loads(text: str) -> dict:
    """Parse a JSON object from model output, tolerating fences and prose."""
    if not text:
        return {}
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        text = m.group(0)
    try:
        return json.loads(text)
    except ValueError:
        return {}


class Run:
    """Everything one incident run touches: cluster, tools, events, usage."""

    def __init__(self, scenario: dict, agent: str, llm: LLM, sink=None, approve=None):
        self.scenario = scenario
        self.agent = agent
        self.llm = llm
        self.cluster = Cluster(scenario)
        self.tb = T.Toolbox(self.cluster)
        self.events: list[dict] = []
        self.sink = sink
        self.approve = approve or auto_approve
        self.usage: dict = {}
        self.pings = 0
        self.model_calls = 0
        self.wall0 = time.time()
        self._series_t = None
        self._grid_last = None
        self.t_alert = None
        self.contract: dict = {}

    # ---------------------------------------------------------------- events
    def emit(self, type_: str, **data):
        # snapshot the payload: later mutations (e.g. contract statuses) must not rewrite history
        data = json.loads(json.dumps(data, default=str))
        ev = {"type": type_, "t": round(self.cluster.t, 1), "clock": clock_str(self.cluster.t),
              "wall": round(time.time() - self.wall0, 2), **data}
        self.events.append(ev)
        if self.sink:
            self.sink(ev)
        return ev

    def tick(self):
        c = self.cluster
        start = self._series_t if self._series_t is not None else max(0.0, c.t - 3600 - (c.t % 60))
        end = c.t - (c.t % 60)
        pts = c.series(start, end, bucket=60.0) if end > start else []
        if pts:
            self._series_t = end
        codes, order = c.grid_codes(c.t)
        grid = None
        if codes != self._grid_last:
            grid, self._grid_last = {"codes": codes, "nodes": order}, codes
        tp = c.throughput(c.t - 120, c.t) or 0.0
        self.emit("tick", points=pts, grid=grid, state=c.job_state(c.t),
                  ratio=round(tp / BASELINE_TPS, 4), step=c.step)

    def say(self, who: str, text: str, mood: str = "talk"):
        if text and text.strip():
            name, role = CAST.get(who, (who, None))
            self.emit("say", who=who, name=name, model=DISPLAY.get(MODELS.get(role, ""), ""),
                      text=text.strip()[:400], mood=mood)

    # ---------------------------------------------------------------- llm
    async def llm_call(self, role: str, who: str, messages, charge: bool = True, **kw) -> dict:
        self.model_calls += 1
        self.emit("thinking", who=who)
        msg = await self.llm.chat(role, messages, usage=self.usage, meter=who, **kw)
        if charge:
            self.cluster.advance(LLM_CALL_S)
        return msg

    async def tool_loop(self, *, role: str, who: str, messages: list, tools: list, handlers: dict,
                        max_rounds: int, stop: set, governor=None, max_tokens: int = 4096,
                        thinking: bool = True):
        """Native tool-calling loop. Read tools in one turn run in parallel."""
        nudges = 0
        for rnd in range(max_rounds):
            msg = await self.llm_call(role, who, messages, tools=tools, max_tokens=max_tokens,
                                      thinking=thinking)
            if msg.get("_finish") == "length" and not msg.get("tool_calls"):
                msg = await self.llm_call(role, who, messages, tools=tools, max_tokens=max_tokens * 2,
                                          thinking=thinking)
            calls = msg.get("tool_calls") or []
            content = (msg.get("content") or "").strip()
            entry = {"role": "assistant", "content": content}
            if calls:
                entry["tool_calls"] = [{"id": tc["id"], "type": "function",
                                        "function": {"name": tc["function"]["name"],
                                                     "arguments": tc["function"].get("arguments") or "{}"}}
                                       for tc in calls]
            messages.append(entry)
            if content:
                self.say(who, content)
            if not calls:
                nudges += 1
                if nudges > 2:
                    return None
                messages.append({"role": "user", "content": "Continue using the tools. Finish by calling "
                                 + " or ".join(sorted(stop)) + "."})
                continue
            batch_cost = 0.0
            result = None
            for tc in calls:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"].get("arguments") or "{}")
                    if not isinstance(args, dict):
                        args = {}
                except ValueError:
                    args = {}
                h = handlers.get(name)
                if h is None:
                    out = f"error: tool {name} is not available to you"
                else:
                    out, cost, fin = await h(args)
                    batch_cost = max(batch_cost, cost)
                    if fin is not None:
                        result = fin
                if governor:
                    out = governor.on_tool(name, args, out)
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": out})
            self.cluster.advance(batch_cost)
            self.tick()
            if result is not None:
                return result
            if governor:
                note = governor.after_round()
                if note == "escalate":
                    return {"escalate": True}
                if note:
                    messages.append({"role": "user", "content": note})
        return None

    def read_handler(self, name: str, who: str):
        async def h(args):
            text, meta = self.tb.run(name, args, who=who, charge=False)
            self.emit("tool", who=who, name=name, args=args, summary=text[:600],
                      highlight=meta.get("highlight"), explained_pct=meta.get("explained_pct"))
            return text, T.COST.get(name, 0), None
        return h


# ===================================================================== Capy
class Governor:
    """Catches loops, stalls and budget overruns in the lead's investigation."""

    def __init__(self, run: Run, budget_min: float, max_calls: int):
        self.run = run
        self.seen: dict[str, float] = {}
        self.calls = 0
        self.max_calls = max_calls
        self.budget_s = budget_min * 60
        self.idle_rounds = 0
        self.new_info = False
        self.warned = False
        self.decisions: list[dict] = []

    def on_tool(self, name, args, out):
        self.calls += 1
        if name in ("note_hypothesis", "propose_diagnosis"):
            self.new_info = True
            return out
        sig = name + json.dumps(args, sort_keys=True)
        if sig in self.seen:
            self.decide("loop", f"repeated {name}({json.dumps(args)[:60]}) from {clock_str(self.seen[sig])}Z")
            return (f"[governor] You already ran this exact call at {clock_str(self.seen[sig])}Z and the answer has not changed. "
                    "Test a different hypothesis or propose a diagnosis.\n" + out)
        self.seen[sig] = self.run.cluster.t
        self.new_info = True
        return out

    def decide(self, kind, reason):
        d = {"decision": kind, "reason": reason}
        self.decisions.append(d)
        self.run.emit("governor", **d)

    def after_round(self):
        spent = self.run.cluster.t - self.run.t_alert
        if self.new_info:
            self.idle_rounds = 0
        else:
            self.idle_rounds += 1
        self.new_info = False
        if spent > self.budget_s + 600 or self.calls > self.max_calls + 10:
            self.decide("escalate", f"investigation budget exhausted ({spent / 60:.0f} min, {self.calls} calls)")
            return "escalate"
        if (spent > self.budget_s or self.calls > self.max_calls) and not self.warned:
            self.warned = True
            self.decide("wrap-up", f"budget reached ({spent / 60:.0f} min, {self.calls} calls)")
            return ("[governor] Budget reached. Propose your best-supported diagnosis now with propose_diagnosis, "
                    "or explain what evidence is missing.")
        if self.idle_rounds >= 2:
            self.idle_rounds = 0
            self.decide("re-plan", "no new evidence for 2 rounds")
            return ("[governor] No new evidence in the last two rounds. Re-plan: list your open hypotheses, pick the "
                    "one with the most explanatory power, and run the single check that would confirm or kill it.")
        return None


CONTRACT_PROMPT = """You are Capy's contract writer. Before anyone investigates, you write the Definition of Done for this incident so the team can check its own work without a human.

Output only JSON:
{"goal": str,
 "done_when": [{"id": "D1", "check": str, "verified_by": "explain_slowdown|skeptic|verify_recovery|logs|ticket"}],
 "constraints": [str],
 "escalate_if": [str],
 "budget": {"max_minutes": int, "max_tool_calls": int}}

Policy floors you must respect (you may be stricter, never looser):
- throughput >= 97% of the 7-day baseline sustained for >= 200 steps after the fix
- the stated root cause(s) explain >= 90% of the slowdown (counterfactual, from rank timings)
- an independent skeptic (different model family) upholds the diagnosis
- no new XID / NCCL WARN / link-down / dataloader-starvation errors during verification
- at most one human ping, only for disruptive actions (drain, restart, pause, kill)
- hardware faults get a Crusoe support ticket with diagnostics attached
Keep it to 5-7 checks, concrete and measurable."""

LEAD_PROMPT = """You are Detective Capy, lead investigator of an AI on-call team that babysits a 512-GPU LLM training run on Crusoe Cloud (CMK cluster, 64x h100-80gb-sxm-ib.8x, TP=8 PP=4 DP=16, synchronous: one slow rank slows everyone).

A throughput or progress alert fired. Find the root cause(s) and PROVE them before anyone touches the cluster.

How you work:
- Field capys already ran a triage sweep (in the first message). Treat it as leads, not conclusions.
- Test hypotheses with read-only tools. Track them with note_hypothesis (open / testing / ruled_out / confirmed).
- A cause is proven only when explain_slowdown shows it removes most of the slowdown; together your causes must explain >= 90%. If your best cause explains much less, keep looking: incidents can have two causes.
- Correlation is not causation: a change that happened before the slowdown is only a suspect if its mechanism fits the evidence (a fleet-wide change cannot explain one slow rank; a localized fault cannot explain a uniform slowdown).
- Crusoe Instance Health only flags hard failures (XID codes). Gray failures (slow but alive) show up in DCGM metrics, counters, logs and rank timings.
- The fix must remove the mechanism of the cause; among fixes that do, choose the least disruptive. Non-disruptive: reset_power_limit, enable_local_cache, revert_rollout. Disruptive (needs the human's approval; costs downtime and a rollback to the last checkpoint): drain_node (already restarts the job with a spare), restart_job (optionally exclude_nodes and/or config_version), pause_job, kill_process. A restart that does not remove the cause fixes nothing.
- You do not execute fixes. When proven, call propose_diagnosis with root_causes, evidence, explained_pct, the fix_plan (ordered actions with exact args) and what you ruled out. An independent skeptic will try to falsify it, then the fix is executed and verified against the definition of done.
- Every tool call costs time while 512 GPUs run slow: batch independent checks in one turn. Before each batch, write one short sentence (max 15 words) saying what you are checking and why.

Definition of done for this incident:
{contract}

Runbook (lessons from past verified incidents):
{brain}"""

SKEPTIC_PROMPT = """You are Inspector Doubt, the independent skeptic on Capy's on-call team. You are deliberately a different model from the lead, so your blind spots differ.

The lead proposes a diagnosis and fix for a problem on a 512-GPU synchronous training job (one slow rank slows all ranks). Try to FALSIFY it before the fix runs:
1. What is the strongest alternative explanation? Check it with at most 5 read-only tool calls (batch them).
2. Does the evidence really support the claimed cause and location?
3. Does the fix plan remove the cause, with exact correct args? Is there a cheaper safe fix (e.g. resetting a power limit instead of draining a node)?

Then call verdict. Uphold only if you could not find a better explanation and the plan is sound. Object with specific evidence (tool results), not vibes. Write one short sentence before each tool batch."""

FIELD_PROMPT = """You are a field capy doing a fast triage sweep ({domain}) for a training-job alert on a 512-GPU H100 cluster. From the tool outputs below, list up to 3 findings that could matter for the alert, most important first. Only report real deviations (compare against the fleet median, the 7-day p50 or the value 30 minutes ago); values that look normal are not findings. Normal ranges: GPU temperature below 85 C with no throttle reasons, SM clock within 3% of the median, shared-disk latency under 5 ms, zero new IB errors, old XIDs from hours ago on other nodes. Severity is high only if it could slow the whole synchronous job right now. Report facts with exact locations (node, gpu, nic). If nothing deviates, return an empty list and headline "nothing unusual".
Output only JSON: {{"findings": [{{"signal": str, "where": str, "severity": "high|medium|low"}}], "headline": str (max 12 words)}}"""

ARCHIVIST_PROMPT = """You are Archivist Capy. Write the case file for this resolved incident and one runbook entry for the team's shared brain. Be concrete and short.
Output only JSON:
{"title": str, "summary": str (2 sentences), "root_cause": str, "proof": str, "fix": str, "verification": str,
 "runbook": {"root_cause_type": str, "signature": str (the telemetry pattern), "confirm_with": str (tool calls that prove it), "fix": str (cheapest safe fix), "pitfalls": str}}"""

NOTE_TOOL = T._fn("note_hypothesis", "Update the case board with your current hypotheses.",
                  {"hypotheses": {"type": "array", "items": {"type": "object", "properties": {
                      "id": {"type": "string"}, "title": {"type": "string"},
                      "status": {"type": "string", "enum": ["open", "testing", "ruled_out", "confirmed"]},
                      "explained_pct": {"type": "number"}, "evidence": {"type": "string"}},
                      "required": ["id", "title", "status"]}}}, ["hypotheses"])
FIX_STEP = {"type": "object", "properties": {
    "action": {"type": "string", "enum": sorted(T.ACTIONS)}, "args": {"type": "object"},
    "why": {"type": "string"}}, "required": ["action", "args"]}
PROPOSE_TOOL = T._fn("propose_diagnosis", "Submit the proven diagnosis and fix plan for skeptic review and execution.",
                     {"root_causes": T.ROOT_CAUSE_SCHEMA,
                      "explained_pct": {"type": "number"},
                      "evidence": {"type": "array", "items": {"type": "string"}},
                      "ruled_out": {"type": "array", "items": {"type": "string"}},
                      "fix_plan": {"type": "array", "items": FIX_STEP}},
                     ["root_causes", "explained_pct", "evidence", "fix_plan"])
VERDICT_TOOL = T._fn("verdict", "Your final judgement on the lead's proposal.",
                     {"upheld": {"type": "boolean"},
                      "objections": {"type": "array", "items": {"type": "string"}},
                      "checks": {"type": "array", "items": {"type": "string"}}}, ["upheld"])


def _int(v, default: int) -> int:
    m = re.search(r"\d+", str(v)) if v is not None and not isinstance(v, bool) else None
    return int(m.group()) if m and int(m.group()) > 0 else default


def _gpu_index(g):
    if isinstance(g, bool):
        return None
    if isinstance(g, float) and g.is_integer():
        g = int(g)
    if isinstance(g, str):
        m = re.fullmatch(r"\s*(?:gpu)?\s*(\d+)\s*", g, re.I)
        g = int(m.group(1)) if m else None
    return g if isinstance(g, int) and 0 <= g < GPUS_PER_NODE else None


def normalize_causes(raw) -> list[dict]:
    """Root causes as the models send them, made well-formed: one node and at
    most one integer GPU per cause. A cause naming several is split up."""
    out = []
    for r in raw or []:
        if not isinstance(r, dict):
            continue
        node = r.get("node")
        nodes = [n.strip() for n in (node if isinstance(node, list) else re.split(r",|\band\b", node) if isinstance(node, str) else [])
                 if isinstance(n, str) and n.strip()] or [None]
        gpu = r.get("gpu")
        gpus = [g for g in map(_gpu_index, gpu if isinstance(gpu, list) else [gpu]) if g is not None] or [None]
        if len(nodes) > 1 or len(gpus) == GPUS_PER_NODE:  # several nodes, or a whole node: node-level causes
            gpus = [None]
        for n in nodes:
            for g in gpus:
                c = {**r, "node": n, "gpu": g}
                out.append({k: v for k, v in c.items() if v is not None})
    return out


def removal_for(rc: dict):
    t = rc.get("type")
    if t in ("gpu_thermal", "gpu_power_cap"):
        if rc.get("node") and rc.get("gpu") is not None:
            return {"kind": "gpu", "node": rc["node"], "gpu": rc["gpu"]}
        if rc.get("node"):
            return {"kind": "node", "node": rc["node"]}
    if t == "cpu_contention" and rc.get("node"):
        return {"kind": "node", "node": rc["node"]}
    if t in ("ib_link", "pcie_link", "nccl_config"):
        return {"kind": "comm"}
    if t == "storage_contention":
        return {"kind": "data_wait"}
    if t == "checkpoint_storage":
        return {"kind": "checkpoint"}
    if t == "gpu_hardware_failure":
        return "hang"
    return None


ACTION_ARGS = {t["function"]["name"]: set(t["function"]["parameters"]["properties"]) for t in T.ACTION_TOOLS}
ARG_ALIASES = {"revert_rollout": {"target": "rollout", "name": "rollout", "daemonset": "rollout", "rollout_name": "rollout",
                                  "component": "rollout"},
               "drain_node": {"node_name": "node", "target": "node"},
               "kill_process": {"process_id": "pid", "process": "pid", "proc": "pid", "node_name": "node"},
               "pause_job": {"name": "job", "job_name": "job"}, "reset_power_limit": {"gpu_index": "gpu", "node_name": "node"}}


def normalize_plan(plan: list) -> list[dict]:
    """Make a fix plan executable: known actions, known args, safe steps first,
    and at most one job restart (a drain already restarts the job)."""
    safe, disruptive, excl, config = [], [], [], None
    for a in plan:
        if not isinstance(a, dict) or a.get("action") not in T.ACTIONS or a["action"] == "file_support_ticket":
            continue
        raw = dict(a.get("args") or {})
        for k, v in list(raw.items()):
            alias = ARG_ALIASES.get(a["action"], {}).get(k)
            if alias and alias not in raw:
                raw[alias] = v
        args = {k: v for k, v in raw.items() if k in ACTION_ARGS[a["action"]]}
        if a["action"] == "revert_rollout" and not args.get("rollout"):
            # no rollout named: a launcher-config revert ({"version": "v42"}) is a restart on that config
            v = raw.get("config_version") or raw.get("version")
            if isinstance(v, str) and re.fullmatch(r"v\d+", v.strip()):
                config = config or v.strip()
            continue
        if a["action"] == "drain_node" and args.get("node"):
            excl.append(args["node"])
        elif a["action"] == "restart_job":
            excl += [n for n in (args.get("exclude_nodes") or []) if isinstance(n, str)]
            config = args.get("config_version") or config
        elif a["action"] in T.DISRUPTIVE:
            disruptive.append({"action": a["action"], "args": args, "why": a.get("why", "")})
        else:
            safe.append({"action": a["action"], "args": args, "why": a.get("why", "")})
    excl = list(dict.fromkeys(excl))
    if len(excl) == 1 and not config:
        disruptive.append({"action": "drain_node", "args": {"node": excl[0], "reason": "proven root cause"}})
    elif excl or config or any(a.get("action") == "restart_job" for a in plan if isinstance(a, dict)):
        args = {}
        if excl:
            args["exclude_nodes"] = excl
        if config:
            args["config_version"] = config
        disruptive.append({"action": "restart_job", "args": args})
    return safe + disruptive


def disruption_cost(c: Cluster, plan: list[dict]) -> dict:
    down, roll = 0.0, 0
    for a in plan:
        if a.get("action") == "drain_node" or (a.get("action") == "restart_job" and (a.get("args") or {}).get("exclude_nodes")):
            down = max(down, SWAP_NODE_S + RESTART_S)
            roll = c.step - c.last_ckpt
        elif a.get("action") == "restart_job":
            down = max(down, RESTART_S)
            roll = c.step - c.last_ckpt
    gpu_h = (down + roll * HEALTHY_STEP_S) * N_GPUS / 3600
    return {"downtime_min": round(down / 60, 1), "rollback_steps": roll, "gpu_hours": round(gpu_h, 1)}


class Capy:
    def __init__(self, llm: LLM, brain: Brain | None = None, learn: bool = True):
        self.llm = llm
        self.brain = brain or Brain()
        self.learn = learn

    async def run(self, scenario: dict, sink=None, approve=None) -> dict:
        run = Run(scenario, "capy", self.llm, sink, approve)
        c = run.cluster
        run.t_alert = c.alert_time()
        c.advance_to(run.t_alert)
        run.emit("start", agent="capy", scenario=scenario["id"], job=sim.JOB,
                 cast={k: {"name": v[0], "model": DISPLAY.get(MODELS.get(v[1], ""), "")} for k, v in CAST.items() if v[1]})
        run.tick()
        run.emit("alert", text=alert_text(c))
        declared, outcome, report, err = [], "escalated", None, None
        try:
            contract = await self.contract(run)
            findings = await self.triage(run)
            declared, outcome, report = await self.investigate_fix_verify(run, contract, findings)
        except LLMError as e:
            run.emit("error", text=str(e))
            outcome, err = "error", str(e)
        return finalize(run, declared, outcome, report, err)

    # ---------------------------------------------------------------- contract
    async def contract(self, run: Run) -> dict:
        c = run.cluster
        status, _ = run.tb.run("job_status", {}, who="contract")
        msg = await run.llm_call("contract", "contract", [
            {"role": "system", "content": CONTRACT_PROMPT},
            {"role": "user", "content": f"Alert: {alert_text(c)}\n\nJob status:\n{status}"}],
            max_tokens=1500, thinking=False, json_mode=True)
        k = _loads(msg.get("content") or "")
        checks = [d if isinstance(d, dict) else {"check": str(d)}
                  for d in (k.get("done_when") if isinstance(k.get("done_when"), list) else [])]
        must = [("throughput", f"throughput >= {int(POLICY['min_ratio'] * 100)}% of baseline for {POLICY['verify_steps']} steps", "verify_recovery"),
                ("explain", f"root cause(s) explain >= {POLICY['min_explained']:.0f}% of the slowdown", "explain_slowdown"),
                ("skeptic", "independent skeptic upholds the diagnosis", "skeptic")]
        text = json.dumps(checks).lower()
        added = []
        for key, check, by in must:
            if key not in text:
                checks.append({"id": f"D{len(checks) + 1}", "check": check + " (policy)", "verified_by": by})
                added.append(key)
        budget = k.get("budget") if isinstance(k.get("budget"), dict) else {}
        k["budget"] = {key: min(_int(budget.get(key), POLICY[key]), POLICY[key]) for key in ("max_minutes", "max_tool_calls")}
        k["done_when"] = [{"id": d.get("id") or f"D{i + 1}", "check": d.get("check", ""), "verified_by": d.get("verified_by", ""),
                           "status": "open"} for i, d in enumerate(checks)]
        k.setdefault("goal", "Restore training throughput to baseline with a proven root cause.")
        run.contract = k
        run.emit("contract", contract=k, policy_added=added)
        run.say("contract", f"Contract written: {len(k['done_when'])} checks must pass before I call this done.")
        return k

    KEYS = {"explain_slowdown": ("explain", "counterfactual", "root cause"),
            "skeptic": ("skeptic", "independent"),
            "verify_recovery": ("throughput", "tok/s", "steps", "xid", "nccl", "error", "loss"),
            "ticket": ("ticket", "support", "diagnostic"),
            "ping": ("ping", "human", "approval", "page")}

    def mark(self, run: Run, kind: str, status: str, note: str = ""):
        for d in run.contract.get("done_when", []):
            text = (d.get("check", "") + " " + d.get("verified_by", "")).lower()
            if d.get("verified_by") == kind or any(k in text for k in self.KEYS.get(kind, ())):
                if d.get("status") == "pass" and status != "pass" and kind == "verify_recovery":
                    continue
                d["status"] = status
                if note:
                    d["note"] = note[:200]
        run.emit("contract_update", done_when=run.contract.get("done_when", []))

    # ---------------------------------------------------------------- triage
    async def triage(self, run: Run) -> list[dict]:
        sweeps = {
            "field-gpu": ("GPUs", [("gpu_outliers", {"metric": "throttle_reasons"}), ("gpu_outliers", {"metric": "sm_clock", "k": 5}),
                                   ("gpu_outliers", {"metric": "temperature", "k": 4}), ("fleet_health", {})]),
            "field-net": ("network", [("network_status", {}), ("rank_timings", {"k": 5}),
                                      ("search_logs", {"pattern": "NCCL WARN|Link down", "minutes": 60})]),
            "field-data": ("data pipeline", [("storage_status", {}), ("search_logs", {"pattern": "prefetch|dataloader", "minutes": 60})]),
            "field-change": ("recent changes", [("change_log", {"minutes": 180}), ("job_config", {}),
                                                ("search_logs", {"pattern": "Xid|Power limit", "minutes": 180})]),
        }
        outputs, costs = {}, {}
        for who, (domain, calls) in sweeps.items():
            parts, cost = [], 0.0
            for name, args in calls:
                text, meta = run.tb.run(name, args, who=who, charge=False)
                cost += T.COST.get(name, 0)
                parts.append(f"$ {name} {json.dumps(args)}\n{text}")
                run.emit("tool", who=who, name=name, args=args, summary=text[:600], highlight=meta.get("highlight"))
            outputs[who], costs[who] = (domain, "\n\n".join(parts)), cost
        run.cluster.advance(max(costs.values()))

        async def summarize(who):
            domain, text = outputs[who]
            try:
                msg = await run.llm_call("field", who, [
                    {"role": "system", "content": FIELD_PROMPT.format(domain=domain)},
                    {"role": "user", "content": text}], max_tokens=2500, thinking=True, json_mode=True, charge=False)
                data = _loads(msg.get("content") or "")
            except LLMError:
                data = {}
            data.setdefault("findings", [])
            run.say(who, data.get("headline") or (data["findings"][0]["signal"] if data["findings"] else "Nothing unusual."))
            return who, domain, data, text

        res = await asyncio.gather(*[summarize(w) for w in sweeps])
        run.cluster.advance(LLM_CALL_S)   # the four field capys think in parallel
        findings = []
        for who, domain, data, text in res:
            findings.append({"who": who, "domain": domain, "findings": data["findings"], "raw": text})
        run.emit("triage", findings=[{"who": f["who"], "domain": f["domain"], "findings": f["findings"]} for f in findings])
        run.tick()
        return findings

    # ---------------------------------------------------------------- main loop
    async def investigate_fix_verify(self, run: Run, contract: dict, findings: list[dict]):
        c = run.cluster
        gov = Governor(run, contract["budget"]["max_minutes"], contract["budget"]["max_tool_calls"])
        board: dict[str, dict] = {}

        async def note(args):
            for h in args.get("hypotheses") or []:
                if isinstance(h, dict) and h.get("id"):
                    board[h["id"]] = {**board.get(h["id"], {}), **h}
            run.emit("hypotheses", items=list(board.values()))
            return "case board updated", 0, None

        async def propose(args):
            return "proposal received; sent to the skeptic", 0, {"proposal": args}

        handlers = {t["function"]["name"]: run.read_handler(t["function"]["name"], "lead") for t in T.READ_TOOLS}
        handlers["note_hypothesis"] = note
        handlers["propose_diagnosis"] = propose
        tools = T.READ_TOOLS + [NOTE_TOOL, PROPOSE_TOOL]
        triage_text = "\n\n".join(
            f"## {f['who']} ({f['domain']})\nfindings: {json.dumps(f['findings'])}\nraw:\n{f['raw']}" for f in findings)
        messages = [
            {"role": "system", "content": LEAD_PROMPT.format(
                contract=json.dumps({k: contract[k] for k in ("goal", "done_when", "constraints", "escalate_if", "budget") if k in contract}, indent=1),
                brain=self.brain.render())},
            {"role": "user", "content": f"Alert: {alert_text(c)}\nTime now: {clock_str(c.t)}Z\n\nTriage sweep:\n{triage_text}"}]
        rejections, attempts = 0, 0
        last_fix = None
        while True:
            res = await run.tool_loop(role="lead", who="lead", messages=messages, tools=tools, handlers=handlers,
                                      max_rounds=14, stop={"propose_diagnosis"}, governor=gov)
            if not res or res.get("escalate"):
                return [], await self.escalate(run, "investigation budget exhausted without a proven cause"), None
            prop = res["proposal"]
            rcs = normalize_causes(prop.get("root_causes"))
            run.emit("proposal", proposal=prop)
            plan = normalize_plan(prop.get("fix_plan") or [])
            # ---- proof gate (deterministic)
            ok, proof = self.proof_gate(run, rcs)
            run.emit("proof", ok=ok, **proof)
            if proof.get("healthy") and last_fix:
                # The job already looks healthy after our fix: re-measure instead of re-proving.
                v = run.tb.verify(POLICY["verify_steps"])
                run.emit("verify", **v)
                self.mark(run, "verify_recovery", "pass" if v["pass"] else "fail", v["summary"])
                if v["pass"]:
                    rcs, prop, proof, verdict = last_fix
                    report = await self.close(run, rcs, prop, proof, verdict, v)
                    return rcs, "resolved", report
                return rcs, await self.escalate(run, f"job healthy but verification fails: {v['summary']}"), None
            if not plan:
                # an executable plan comes first, so the skeptic reviews something real
                messages.append({"role": "user", "content": "[fixer] Your proposal has no executable fix_plan. Propose again with concrete actions and exact args."})
                continue
            self.mark(run, "explain_slowdown", "pass" if ok else "fail", proof["summary"])
            if not ok:
                rejections += 1
                if rejections > 3:
                    return rcs, await self.escalate(run, "could not prove a root cause that explains the slowdown"), None
                messages.append({"role": "user", "content": f"[proof gate] REJECTED: {proof['summary']} The contract requires >= {POLICY['min_explained']:.0f}%. Keep investigating (maybe a second cause), then propose again."})
                continue
            # ---- skeptic
            verdict = await self.skeptic(run, prop, proof)
            self.mark(run, "skeptic", "pass" if verdict["upheld"] else "fail", "; ".join(verdict.get("objections") or [])[:200])
            if not verdict["upheld"]:
                rejections += 1
                if rejections > 3:
                    return rcs, await self.escalate(run, "lead and skeptic disagree; human judgement needed"), None
                messages.append({"role": "user", "content": "[skeptic] OBJECTION: " + " | ".join(verdict.get("objections") or ["no reason given"])
                                 + "\nAddress these with evidence, then propose again (or a revised diagnosis/fix)."})
                continue
            # ---- fix
            approved = await self.apply_fix(run, rcs, prop, proof, verdict, plan)
            last_fix = (rcs, prop, proof, verdict)
            if not approved:
                return rcs, await self.escalate(run, "human denied the disruptive fix"), None
            # ---- verify
            run.say("lead", f"Fix applied. Verifying: {POLICY['verify_steps']} steps at >= {int(POLICY['min_ratio'] * 100)}% of baseline.")
            run.emit("verify_start", steps=POLICY["verify_steps"])
            v = run.tb.verify(POLICY["verify_steps"])
            run.emit("verify", **v)
            run.tick()
            self.mark(run, "verify_recovery", "pass" if v["pass"] else "fail", v["summary"])
            if v["pass"]:
                report = await self.close(run, rcs, prop, proof, verdict, v)
                return rcs, "resolved", report
            attempts += 1
            if attempts >= 2:
                return rcs, await self.escalate(run, f"fix verified FAILED twice: {v['summary']}"), None
            run.say("governor", "Verification failed. Re-planning instead of declaring victory.", mood="alarm")
            gov.decide("re-plan", "verification failed: " + v["summary"])
            messages.append({"role": "user", "content": f"[verifier] Verification FAILED after your fix: {v['summary']}. "
                             "The problem (or part of it) remains. Re-investigate from current telemetry and propose again."})

    async def close(self, run: Run, rcs, prop, proof, verdict, v) -> dict:
        if not any(r.get("type") in HARDWARE_TYPES for r in rcs):
            self.mark(run, "ticket", "pass", "n/a: no hardware fault")
        ticketed = set()
        for r in rcs:
            if r.get("type") in HARDWARE_TYPES and r.get("node") and r["node"] not in ticketed:
                ticketed.add(r["node"])
                text, _ = run.tb.run("file_support_ticket", {"node": r["node"], "summary": f"{r['type']}: {r.get('detail', '')}"[:200]}, who="lead")
                run.emit("ticket", text=text, node=r["node"])
                self.mark(run, "ticket", "pass")
        self.mark(run, "ping", "pass" if run.pings <= POLICY["max_pings"] else "fail", f"{run.pings} ping(s)")
        return await self.archive(run, rcs, prop, proof, verdict, v)

    def proof_gate(self, run: Run, rcs: list[dict]):
        c = run.cluster
        if not rcs:
            return False, {"explained_pct": 0, "summary": "no root causes given"}
        removes, hang = [], None
        for r in rcs:
            rm = removal_for(r)
            if rm == "hang":
                hang = r
            elif rm:
                removes.append(rm)
        if hang and c.job_state(c.t) in ("hung", "crashed"):
            rt = c.rank_timings(c.t)
            h = rt.get("hang") or {}
            ok = h.get("node") == hang.get("node")
            return ok, {"explained_pct": 100.0 if ok else 0.0,
                        "summary": (f"job is hung; rank {h.get('rank_without_heartbeat')} on {h.get('node')} gpu{h.get('gpu')} has no heartbeat"
                                    + ("; matches the claimed node" if ok else f"; claimed {hang.get('node')} does not match"))}
        if not removes:
            return False, {"explained_pct": 0, "summary": "root causes could not be mapped to a measurable effect"}
        res = c.explain(c.t, removes)
        if "error" in res:
            return False, {"explained_pct": 0, "summary": res["error"]}
        if res["current_step_s"] <= sim.HEALTHY_STEP_S * 1.03:
            return False, {"explained_pct": 0, "healthy": True,
                           "summary": f"no slowdown left to explain: current step {res['current_step_s']}s vs healthy {res['healthy_step_s']}s"}
        pct = res["explained_pct"]
        return pct >= POLICY["min_explained"], {"explained_pct": pct, "removes": removes,
                                                "summary": f"counterfactual: removing {json.dumps(removes)} explains {pct}% of the slowdown"}

    async def skeptic(self, run: Run, prop: dict, proof: dict) -> dict:
        async def verdict(args):
            return "verdict recorded", 0, {"verdict": args}

        handlers = {t["function"]["name"]: run.read_handler(t["function"]["name"], "skeptic") for t in T.READ_TOOLS}
        handlers["verdict"] = verdict
        messages = [{"role": "system", "content": SKEPTIC_PROMPT},
                    {"role": "user", "content": f"Alert: {alert_text(run.cluster)}\nTime now: {clock_str(run.cluster.t)}Z\n"
                                                f"Lead's proposal:\n{json.dumps(prop, indent=1)}\n\nProof gate: {proof['summary']}"}]
        res = None
        try:
            res = await run.tool_loop(role="skeptic", who="skeptic", messages=messages, tools=T.READ_TOOLS + [VERDICT_TOOL],
                                      handlers=handlers, max_rounds=4, stop={"verdict"}, max_tokens=8192)
            if not res:
                messages.append({"role": "user", "content": "Time is up. Call verdict now with your judgement."})
                msg = await run.llm_call("skeptic", "skeptic", messages, tools=[VERDICT_TOOL], thinking=False,
                                         tool_choice={"type": "function", "function": {"name": "verdict"}}, max_tokens=2048)
                for tc in msg.get("tool_calls") or []:
                    if tc["function"]["name"] == "verdict":
                        res = {"verdict": _loads(tc["function"].get("arguments") or "{}")}
        except LLMError as e:
            run.emit("error", text=f"skeptic unavailable: {e}")
        out = dict((res or {}).get("verdict") or {})
        if not out:
            out = {"upheld": True, "objections": [], "checks": ["no verdict returned; falling back to the proof gate"]}
        for key in ("objections", "checks"):
            v = out.get(key) or []
            if isinstance(v, str):
                parsed = _loads('{"x": ' + v + "}").get("x") if v.strip().startswith("[") else None
                v = parsed if isinstance(parsed, list) else [v]
            out[key] = [str(x) for x in v]
        out["upheld"] = bool(out.get("upheld"))
        run.emit("skeptic", **out)
        run.say("skeptic", ("Upheld. " + "; ".join(out["checks"][:2])) if out["upheld"]
                else ("Objection! " + "; ".join(out["objections"][:2])), mood="happy" if out["upheld"] else "alarm")
        return out

    async def apply_fix(self, run: Run, rcs, prop, proof, verdict, plan) -> bool:
        c = run.cluster
        disruptive = [a for a in plan if a["action"] in T.DISRUPTIVE]
        if disruptive:
            cost = disruption_cost(c, plan)
            causes = "; ".join(r.get("type", "?") + ("" if r.get("type") in FLEET_TYPES or not r.get("node") else f" on {r['node']}")
                               for r in rcs)
            req = {"title": "Approve fix for " + sim.JOB,
                   "root_causes": rcs, "explained_pct": proof["explained_pct"],
                   "evidence": (prop.get("evidence") or [])[:4],
                   "skeptic": "upheld" if verdict["upheld"] else "objected",
                   "plan": plan, "cost": cost,
                   "text": (f"{causes} explains {proof['explained_pct']:.0f}% of the slowdown. "
                            f"Plan: {', '.join(a['action'] for a in plan)}. Cost: ~{cost['downtime_min']} min downtime, "
                            f"{cost['rollback_steps']} steps rollback (~{cost['gpu_hours']} GPU-h).")}
            run.pings += 1
            run.emit("ping", request=req)
            approved = await run.approve(req, run)
            run.emit("ping_response", approved=bool(approved))
            if not approved:
                return False
        for a in plan:
            res = run.tb.act(a["action"], a.get("args") or {}, who="capy")
            run.emit("action", name=a["action"], args=a.get("args") or {}, result=res[:300],
                     disruptive=a["action"] in T.DISRUPTIVE)
            run.tick()
        return True

    async def archive(self, run, rcs, prop, proof, verdict, v) -> dict:
        tl = [f"{e['clock']} {e['type']}: " + (e.get("text") or e.get("name") or e.get("summary") or "")[:120]
              for e in run.events if e["type"] in ("alert", "proposal", "proof", "skeptic", "ping", "action", "verify")]
        try:
            msg = await run.llm_call("archivist", "archivist", [
                {"role": "system", "content": ARCHIVIST_PROMPT},
                {"role": "user", "content": json.dumps({"root_causes": rcs, "evidence": prop.get("evidence"),
                                                        "proof": proof["summary"], "skeptic": verdict,
                                                        "fix_plan": prop.get("fix_plan"), "verification": v["summary"],
                                                        "timeline": tl})}],
                max_tokens=1500, thinking=False, json_mode=True)
            rep = _loads(msg.get("content") or "")
        except LLMError:
            rep = {}
        rep.setdefault("title", "Incident resolved")
        rb = rep.get("runbook")
        if self.learn and isinstance(rb, dict) and rb.get("signature"):
            if self.brain.add(rb):
                run.emit("brain", entry=rb)
        run.emit("closed", report=rep)
        run.say("archivist", rep.get("summary") or "Case closed.", mood="happy")
        return rep

    async def escalate(self, run: Run, reason: str) -> str:
        run.pings += 1
        run.emit("escalate", reason=reason)
        run.say("governor", f"Escalating to the human: {reason}", mood="alarm")
        return "escalated"


# ===================================================================== baseline
BASELINE_PROMPT = """You are an on-call SRE agent for an LLM pre-training job running on a 512-GPU Crusoe H100 cluster (CMK, 64x h100-80gb-sxm-ib.8x, TP=8 PP=4 DP=16).
An alert fired. Investigate with the tools, fix the problem, and call declare_resolved with the root cause(s) once the job is healthy again.
Disruptive actions (drain_node, restart_job, pause_job, kill_process) page the human for approval. Be efficient: the cluster is losing GPU-hours while you work."""


async def run_baseline(scenario: dict, llm: LLM, sink=None, approve=None) -> dict:
    run = Run(scenario, "baseline", llm, sink, approve)
    c = run.cluster
    run.t_alert = c.alert_time()
    c.advance_to(run.t_alert)
    run.emit("start", agent="baseline", scenario=scenario["id"], job=sim.JOB)
    run.tick()
    run.emit("alert", text=alert_text(c))
    handlers = {t["function"]["name"]: run.read_handler(t["function"]["name"], "baseline") for t in T.READ_TOOLS}

    def act_handler(name):
        async def h(args):
            if name in T.DISRUPTIVE:
                req = {"title": f"Approve {name}", "plan": [{"action": name, "args": args}],
                       "text": f"Agent requests {name} {json.dumps(args)}"}
                run.pings += 1
                run.emit("ping", request=req)
                ok = await run.approve(req, run)
                run.emit("ping_response", approved=bool(ok))
                if not ok:
                    return "denied by human", 0, None
            res = run.tb.act(name, args, who="baseline")
            run.emit("action", name=name, args=args, result=res[:300], disruptive=name in T.DISRUPTIVE)
            return res, 0, None
        return h

    for t in T.ACTION_TOOLS:
        handlers[t["function"]["name"]] = act_handler(t["function"]["name"])

    async def verify(args):
        v = run.tb.verify(args.get("steps") or 200)
        run.emit("verify", **v)
        return v["summary"], 0, None

    async def declare(args):
        return "declared", 0, {"declare": args}

    handlers["verify_recovery"] = verify
    handlers["declare_resolved"] = declare
    messages = [{"role": "system", "content": BASELINE_PROMPT},
                {"role": "user", "content": f"Alert: {alert_text(c)}\nTime now: {clock_str(c.t)}Z"}]
    declared, outcome, err = [], "gave_up", None
    try:
        res = await run.tool_loop(role="baseline", who="baseline", messages=messages,
                                  tools=T.READ_TOOLS + T.ACTION_TOOLS + [T.VERIFY_TOOL, T.DECLARE_TOOL],
                                  handlers=handlers, max_rounds=24, stop={"declare_resolved"})
        if res and "declare" in res:
            d = res["declare"]
            declared = normalize_causes(d.get("root_causes"))
            outcome = "resolved"
            run.emit("closed", report={"title": "Declared resolved", "summary": d.get("summary", ""),
                                       "fix": d.get("fix_applied", "")})
            run.say("baseline", "Resolved: " + (d.get("summary") or ""))
    except LLMError as e:
        run.emit("error", text=str(e))
        outcome, err = "error", str(e)
    return finalize(run, declared, outcome, None, err)


# ===================================================================== scoring
def match_causes(declared: list[dict], gt: list[dict]) -> tuple[int, int]:
    hit = 0
    for g in gt:
        for d in declared:
            if d.get("type") != g["type"]:
                continue
            if g.get("node") and d.get("node") != g["node"]:
                continue
            hit += 1
            break
    return hit, len(gt)


def finalize(run: Run, declared: list[dict], outcome: str, report: dict | None, error: str | None = None) -> dict:
    c = run.cluster
    t_done = c.t
    horizon = max(run.t_alert + HORIZON_S, c.t + 600)
    c.advance_to(horizon)
    tp = c.throughput(c.t - 600, c.t) or 0.0
    ratio = tp / BASELINE_TPS
    fixed = ratio >= POLICY["min_ratio"] and c.job_state(c.t) == "running"
    gt = c.ground_truth()
    hit, n = match_causes(declared, gt)
    disruptive = [a for a in c.actions if a["name"] in T.DISRUPTIVE and a.get("ok")]
    wasted = [a for a in disruptive if not a.get("fixed")]
    claimed = outcome == "resolved"
    totals = defaultdict(float)
    for u in run.usage.values():
        for k, v in u.items():
            if isinstance(v, (int, float)):
                totals[k] += v
    result = {
        "scenario": run.scenario["id"], "kinds": run.scenario.get("kinds"), "split": run.scenario.get("split"),
        "heldout": run.scenario.get("heldout", False), "agent": run.agent, "outcome": outcome,
        "claimed_done": claimed, "fixed": fixed, "false_done": claimed and not fixed,
        "diagnosis_correct": hit == n and n > 0, "causes_matched": hit, "causes_total": n,
        "declared": declared, "ground_truth": gt,
        "final_ratio": round(ratio, 4), "gpu_hours_lost": round(c.gpu_hours_lost(), 1),
        "minutes_to_done": round((t_done - run.t_alert) / 60, 1),
        "disruptive_actions": len(disruptive), "wasted_disruptive_actions": len(wasted),
        "pings": run.pings, "tool_calls": len(run.tb.calls), "model_calls": run.model_calls,
        "prompt_tokens": int(totals["prompt_tokens"]), "completion_tokens": int(totals["completion_tokens"]),
        "cached_tokens": int(totals["cached_tokens"]), "cost_usd": round(totals["cost_usd"], 4),
        "wall_s": round(time.time() - run.wall0, 1),
        "usage": {k: {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in u.items()} for k, u in run.usage.items()},
        "report": report,
    }
    if error:
        result["error"] = error
    run.emit("result", result={k: v for k, v in result.items() if k not in ("usage",)})
    return result
