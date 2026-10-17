"""Agent-facing tools over the simulated cluster.

Read-only tools mirror what Crusoe exposes in production (Command Center
metrics via PromQL, JournalD/NCCL logs via LogsQL, Instance Health, the
audit log, `crusoe diagnostics vm create` bug reports) plus a PyTorch-style
flight recorder. Each call charges simulated time, so slow investigations
cost GPU-hours exactly like they would at 2 a.m.
"""
from __future__ import annotations

import json
import statistics

from . import sim
from .sim import (BASELINE_TPS, GPUS_PER_NODE, HEALTHY_STEP_S, JOB, TOKENS_PER_STEP, Cluster,
                  clock_str, decode_throttle)

ROOT_CAUSE_TYPES = {
    "gpu_thermal": "GPU overheating / cooling fault causing thermal clock throttling",
    "gpu_power_cap": "GPU power limit set below TDP causing power-capped clocks",
    "gpu_hardware_failure": "GPU hard failure (XID error, fell off the bus, ECC)",
    "ib_link": "InfiniBand link / NIC errors, flapping or degraded",
    "pcie_link": "PCIe link degraded (width or generation) or AER errors",
    "nccl_config": "NCCL / launcher configuration regression",
    "storage_contention": "shared storage contention starving the dataloader",
    "checkpoint_storage": "slow checkpoint saves (storage tier or path)",
    "cpu_contention": "host CPU contention from another process",
    "other": "anything else",
}

METRICS = {
    "sm_clock": ("DCGM_FI_DEV_SM_CLOCK", "MHz", "low"),
    "temperature": ("DCGM_FI_DEV_GPU_TEMP", "C", "high"),
    "power": ("DCGM_FI_DEV_POWER_USAGE", "W", "low"),
    "power_limit": ("DCGM_FI_DEV_POWER_MGMT_LIMIT", "W", "low"),
    "sm_active": ("DCGM_FI_PROF_SM_ACTIVE", "", "both"),
    "gpu_util": ("DCGM_FI_DEV_GPU_UTIL", "%", "low"),
    "pcie_link_width": ("DCGM_FI_DEV_PCIE_LINK_WIDTH", "lanes", "low"),
    "pcie_replay": ("DCGM_FI_DEV_PCIE_REPLAY_COUNTER", "", "high"),
    "throttle_reasons": ("DCGM_FI_DEV_CLOCK_THROTTLE_REASONS", "", "flags"),
    "xid": ("DCGM_FI_DEV_XID_ERRORS", "", "flags"),
}

# simulated seconds charged per call
COST = {
    "job_status": 5, "fleet_health": 8, "gpu_outliers": 20, "gpu_detail": 8,
    "node_detail": 10, "rank_timings": 15, "explain_slowdown": 10, "search_logs": 15,
    "change_log": 8, "network_status": 15, "storage_status": 10, "job_config": 5,
    "collect_diagnostics": 45, "file_support_ticket": 5,
}

DISRUPTIVE = {"drain_node", "restart_job", "pause_job", "kill_process"}
SAFE_ACTIONS = {"reset_power_limit", "enable_local_cache", "revert_rollout", "file_support_ticket"}
ACTIONS = DISRUPTIVE | SAFE_ACTIONS


def _fn(name, desc, props=None, required=None):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props or {}, "required": required or []}}}


NODE = {"type": "string", "description": "node name, e.g. np-9addff51-17"}
GPU = {"type": "integer", "minimum": 0, "maximum": 7}

READ_TOOLS = [
    _fn("job_status", "Training job status: state, step, throughput vs 7-day baseline, step time, loss, last checkpoint, config version."),
    _fn("fleet_health", "Crusoe Instance Health for every node (Healthy/Degraded/Unhealthy with XID codes), job members and spares."),
    _fn("gpu_outliers", "Rank all in-job GPUs by one DCGM metric over the last 5 minutes and return the most anomalous ones with the fleet median.",
        {"metric": {"type": "string", "enum": list(METRICS)}, "k": {"type": "integer", "minimum": 1, "maximum": 16}}, ["metric"]),
    _fn("gpu_detail", "All DCGM fields for one GPU now and 30 minutes ago, with throttle reasons decoded.",
        {"node": NODE, "gpu": GPU}, ["node", "gpu"]),
    _fn("node_detail", "One node: health, ranks, CPU, top processes, per-NIC InfiniBand counters, per-GPU summary.",
        {"node": NODE}, ["node"]),
    _fn("rank_timings", "Flight recorder: per-rank compute / comm-wait / data-wait for recent steps, the critical path, slowest ranks and links. Shows the stuck collective if the job is hung.",
        {"k": {"type": "integer", "minimum": 1, "maximum": 16}}),
    _fn("explain_slowdown", "Counterfactual check from observed rank timings: what share of the slowdown disappears if the given effects were removed. Use it to prove a root cause explains the slowdown.",
        {"remove": {"type": "array", "items": {"type": "object", "properties": {
            "kind": {"type": "string", "enum": ["gpu", "node", "ranks", "comm", "data_wait", "checkpoint"]},
            "node": {"type": "string"}, "gpu": {"type": "integer"},
            "ranks": {"type": "array", "items": {"type": "integer"}}}, "required": ["kind"]}}},
        ["remove"]),
    _fn("search_logs", "Search JournalD (kernel, nvidia-smi), NCCL and trainer logs with a regex.",
        {"pattern": {"type": "string"}, "node": NODE, "minutes": {"type": "integer", "minimum": 1, "maximum": 720},
         "limit": {"type": "integer", "minimum": 1, "maximum": 40}}, ["pattern"]),
    _fn("change_log", "Audit log of config rollouts, job launches, restarts, ssh sessions and daemonset rollouts.",
        {"minutes": {"type": "integer", "minimum": 1, "maximum": 1440}}),
    _fn("network_status", "InfiniBand fabric: per-node error rates and link downs, NIC usage, NCCL bus bandwidth, IB devices in use."),
    _fn("storage_status", "Shared Disk read throughput, latency and top clients for the training data volume; dataloader wait."),
    _fn("job_config", "Launcher config (NCCL and trainer env) in effect, and the diff between known versions.",
        {"version": {"type": "string"}}),
    _fn("collect_diagnostics", "Crusoe diagnostics bug report for a node (nvidia-smi -q excerpt + kernel XID log). Slow: ~45 s.",
        {"node": NODE}, ["node"]),
]

ACTION_TOOLS = [
    _fn("reset_power_limit", "Non-disruptive: reset GPU power limit to 700 W (one GPU, or all GPUs of the node if gpu omitted).",
        {"node": NODE, "gpu": GPU}, ["node"]),
    _fn("enable_local_cache", "Non-disruptive: stage upcoming data shards on local NVMe (~3 min warm-up), bypassing shared storage."),
    _fn("revert_rollout", "Non-disruptive: roll back a daemonset or config rollout by name (the target shown in change_log, e.g. a daemonset). Does not undo state it already applied on nodes.",
        {"rollout": {"type": "string", "description": "daemonset / rollout name from change_log"}}, ["rollout"]),
    _fn("file_support_ticket", "File a Crusoe support ticket with the node's diagnostics attached.",
        {"node": NODE, "summary": {"type": "string"}}, ["node", "summary"]),
    _fn("drain_node", "DISRUPTIVE: cordon the node, swap in a hot spare and restart the job from the last checkpoint (~7.5 min downtime + rollback). This already restarts the job.",
        {"node": NODE, "reason": {"type": "string"}}, ["node"]),
    _fn("restart_job", "DISRUPTIVE: restart the job from the last checkpoint (~3.5 min + rollback), optionally excluding nodes (swapped for spares) and/or with a different launcher config version.",
        {"exclude_nodes": {"type": "array", "items": {"type": "string"}}, "config_version": {"type": "string"}}),
    _fn("pause_job", "DISRUPTIVE (affects another team): pause another job by name.",
        {"job": {"type": "string"}}, ["job"]),
    _fn("kill_process", "DISRUPTIVE: kill a process on a node.", {"node": NODE, "pid": {"type": "integer"}}, ["node", "pid"]),
]

VERIFY_TOOL = _fn("verify_recovery", "Wait for N training steps (default 200) and report throughput vs baseline, step time, new errors and loss trend for exactly those steps.",
                  {"steps": {"type": "integer", "minimum": 20, "maximum": 400}})

ROOT_CAUSE_SCHEMA = {"type": "array", "items": {"type": "object", "properties": {
    "type": {"type": "string", "enum": list(ROOT_CAUSE_TYPES)},
    "node": {"type": "string"}, "gpu": {"type": "integer"}, "nic": {"type": "string"},
    "detail": {"type": "string"}}, "required": ["type"]}}

DECLARE_TOOL = _fn("declare_resolved", "Finish the incident. Call once, when you believe the job is fixed.",
                   {"root_causes": ROOT_CAUSE_SCHEMA, "summary": {"type": "string"},
                    "fix_applied": {"type": "string"}}, ["root_causes", "summary"])


def _num(v: float) -> str:
    a = abs(v)
    return f"{v:.0f}" if a >= 100 else f"{v:.1f}" if a >= 1 else f"{v:.3f}"


class Toolbox:
    """Executes tool calls against one Cluster and formats compact results."""

    def __init__(self, cluster: Cluster):
        self.c = cluster
        self.calls: list[dict] = []

    # ---------------------------------------------------------------- dispatch
    def run(self, name: str, args: dict | None, who: str = "agent", charge: bool = True) -> tuple[str, dict]:
        args = dict(args or {})
        fn = getattr(self, f"t_{name}", None)
        if fn is None:
            return f"error: unknown tool {name}", {}
        if charge:
            self.c.advance(COST.get(name, 0))
        try:
            text, meta = fn(**args)
        except TypeError as e:
            text, meta = f"error: bad arguments for {name}: {e}", {}
        except KeyError as e:
            text, meta = f"error: unknown {e}", {}
        self.calls.append({"t": self.c.t, "who": who, "name": name, "args": args})
        return text, meta

    # ---------------------------------------------------------------- helpers
    def _member_gpus(self, t):
        return [(n, g) for n in self.c.members(t).values() for g in range(GPUS_PER_NODE)]

    def _window_mean(self, node, gpu, field, t, minutes=5, every=30):
        vals = []
        k = 0
        while k <= minutes * 60:
            smp = self.c.gpu_sample(node, gpu, t - k)
            if smp is not None:
                vals.append(smp[field])
            k += every
        return (sum(vals) / len(vals)) if vals else None

    def _rank(self, node, gpu, t):
        r = self.c.rank_of(node, gpu, t)
        return f"rank{r}" if r is not None else "not-in-job"

    # ---------------------------------------------------------------- read tools
    def t_job_status(self):
        c, t = self.c, self.c.t
        state = c.job_state(t)
        ver, cfg = c.config_at(t)
        tp = c.throughput(t - 300, t) or 0.0
        recent = c.recent_steps(t, 300)
        p50 = statistics.median([s[2] for s in recent]) if recent else None
        last_ck = c.ckpts[-1] if c.ckpts else None
        lines = [f"job {JOB} on CMK cluster {sim.CLUSTER} ({sim.LOCATION}): 64x {sim.INSTANCE_TYPE} = 512 GPUs, {cfg['PARALLELISM']}",
                 f"now {clock_str(t)}Z | state: {state} | step {c.step}"]
        if state == "running":
            lines.append(f"throughput (5m): {tp / 1000:.1f}k tok/s = {100 * tp / BASELINE_TPS:.1f}% of 7-day p50 {BASELINE_TPS / 1000:.1f}k tok/s"
                         + (f" | step_time p50 {p50:.2f}s (7-day p50 {sim.BASE_STEP_S:.2f}s)" if p50 else ""))
        else:
            last = c.steps[-1] if c.steps else None
            lines.append(f"no training progress; last completed step {last[1] if last else '?'} at {clock_str(last[0]) if last else '?'}Z")
        lines.append(f"loss {c.loss(c.step):.4f} (trend normal) | launcher config {ver}"
                     + (f" | last checkpoint step_{last_ck[1]} at {clock_str(last_ck[0])}Z (save {last_ck[2]:.1f}s)" if last_ck else ""))
        downs = [d for d in c.down if d[1] <= t]
        if downs:
            a, b, why = downs[-1]
            lines.append(f"last (re)launch {clock_str(b)}Z ({why})")
        return "\n".join(lines), {"state": state, "tps": tp}

    def t_fleet_health(self):
        c, t = self.c, self.c.t
        mem = c.members(t)
        bad, ok = [], 0
        for node in mem.values():
            st, codes = c.instance_health(node, t)
            if st == "Healthy":
                ok += 1
            else:
                bad.append(f"  {node} (in job): {st} [{', '.join(codes)}]")
        spares = []
        for node in c.nodes:
            if node in mem.values():
                continue
            st, codes = c.instance_health(node, t)
            tag = "cordoned" if node in c.cordoned else "hot spare"
            spares.append(f"  {node} ({tag}): {st}" + (f" [{', '.join(codes)}]" if codes else ""))
        text = (f"Crusoe Instance Health (Command Center), {len(c.nodes)} nodes:\n  in job: {ok} Healthy"
                + (f", {len(bad)} not healthy" if bad else "") + "\n" + "\n".join(bad + spares))
        return text, {}

    def t_gpu_outliers(self, metric, k=8):
        c, t = self.c, self.c.t
        if metric not in METRICS:
            return f"error: metric must be one of {list(METRICS)}", {}
        field, unit, bad_dir = METRICS[metric]
        k = max(1, min(int(k), 16))
        rows, missing = [], []
        for node, g in self._member_gpus(t):
            if bad_dir == "flags":
                smp = c.gpu_sample(node, g, t)
                if smp is None:
                    missing.append((node, g))
                    continue
                rows.append((node, g, smp[field]))
            else:
                v = self._window_mean(node, g, field, t)
                if v is None:
                    missing.append((node, g))
                    continue
                rows.append((node, g, v))
        out = []
        if bad_dir == "flags":
            hits = [r for r in rows if r[2]]
            if metric == "throttle_reasons":
                hits = [r for r in hits if r[2] & ~0x01]
                fmt = lambda v: "|".join(decode_throttle(int(v)))
            else:
                fmt = lambda v: f"XID {int(v)} (last recorded)"
            out.append(f"{field} on {len(rows)} readable in-job GPUs: {len(hits)} non-zero")
            for node, g, v in hits[:k]:
                out.append(f"  {node} gpu{g} {self._rank(node, g, t)}: {fmt(v)}")
        else:
            vals = sorted(r[2] for r in rows)
            med = vals[len(vals) // 2] if vals else 0
            p5, p95 = (vals[int(0.05 * len(vals))], vals[int(0.95 * len(vals)) - 1]) if vals else (0, 0)
            if bad_dir == "low":
                order = sorted(rows, key=lambda r: r[2])
            elif bad_dir == "high":
                order = sorted(rows, key=lambda r: -r[2])
            else:
                order = sorted(rows, key=lambda r: -abs(r[2] - med))
            out.append(f"{field} ({unit}), 5-min mean over {len(rows)} readable in-job GPUs: median {_num(med)}, p5 {_num(p5)}, p95 {_num(p95)}")
            out.append(f"most anomalous {k}:")
            for node, g, v in order[:k]:
                dev = (v - med) / med * 100 if med else 0
                out.append(f"  {node} gpu{g} {self._rank(node, g, t)}: {_num(v)}{unit} ({dev:+.1f}% vs median)")
        if missing:
            out.append(f"NO DATA (DCGM cannot read): " + ", ".join(f"{n} gpu{g}" for n, g in missing))
        if bad_dir == "flags":
            hl = [{"node": n, "gpu": g} for n, g, _ in hits[:4]]
        else:
            def notable(v):
                if not med:
                    return False
                rel = (v - med) / med
                return rel < -0.05 if bad_dir == "low" else rel > 0.15 if bad_dir == "high" else abs(rel) > 0.15
            hl = [{"node": n, "gpu": g} for n, g, v in order[:4] if notable(v)]
        hl += [{"node": n, "gpu": g} for n, g in missing[:2]]
        return "\n".join(out), {"highlight": hl}

    def t_gpu_detail(self, node, gpu):
        c, t = self.c, self.c.t
        gpu = int(gpu)
        if node not in c._nidx:
            return f"error: unknown node {node}", {}
        now, before = c.gpu_sample(node, gpu, t), c.gpu_sample(node, gpu, t - 1800)
        head = f"{node} gpu{gpu} (PCI 00000000:{sim.GPU_BUS[gpu]}:00.0) {self._rank(node, gpu, t)}"
        if now is None:
            return head + ": NO DATA - DCGM cannot communicate with this GPU", {"highlight": [{"node": node, "gpu": gpu}]}
        lines = [head, f"{'field':44s} {'now':>10s} {'30m ago':>10s}"]
        for kf, v in now.items():
            b = before.get(kf) if before else None
            if kf == "DCGM_FI_DEV_CLOCK_THROTTLE_REASONS":
                lines.append(f"{kf:44s} {hex(v):>10s} {hex(b) if b is not None else '-':>10s}  -> now: {'|'.join(decode_throttle(v))}")
            else:
                lines.append(f"{kf:44s} {v!s:>10s} {b if b is not None else '-'!s:>10s}")
        return "\n".join(lines), {"highlight": [{"node": node, "gpu": gpu}]}

    def t_node_detail(self, node):
        c, t = self.c, self.c.t
        if node not in c._nidx:
            return f"error: unknown node {node}", {}
        st, codes = c.instance_health(node, t)
        slot = c.slot_of(node, t)
        ns = c.node_sample(node, t)
        lines = [f"{node} ({sim.INSTANCE_TYPE}, {sim.node_ip(c._nidx[node])}): Instance Health {st}"
                 + (f" [{', '.join(codes)}]" if codes else "")
                 + (f" | job slot {slot}, ranks {slot * 8}-{slot * 8 + 7}" if slot is not None else " | not in job"),
                 f"CPU {ns['cpu_util_pct']}% of {ns['vcpus']} vCPU | mem {ns['mem_used_pct']}%",
                 "top processes: " + "; ".join(f"pid {p['pid']} {p['user']} {p['cpu_pct']}% {p['cmd']}" for p in ns["top_processes"]),
                 "GPUs: " + "; ".join(self._gpu_brief(node, g, t) for g in range(GPUS_PER_NODE)),
                 "InfiniBand (counters since boot; delta = last 10 min):"]
        for h in range(GPUS_PER_NODE):
            a, b = c.ib_sample(node, h, t), c.ib_sample(node, h, t - 600)
            lines.append(f"  mlx5_{h} {a['state']:6s} {a['rate_gbps']:6.1f} Gb/s | symbol_error {a['symbol_error']} (+{a['symbol_error'] - b['symbol_error']})"
                         f" link_downed {a['link_downed']} (+{a['link_downed'] - b['link_downed']}) rcv_errors {a['port_rcv_errors']} (+{a['port_rcv_errors'] - b['port_rcv_errors']})")
        return "\n".join(lines), {"highlight": [{"node": node}]}

    def _gpu_brief(self, node, g, t):
        s = self.c.gpu_sample(node, g, t)
        if s is None:
            return f"gpu{g} NO DATA"
        thr = s["DCGM_FI_DEV_CLOCK_THROTTLE_REASONS"] & ~0x01
        return (f"gpu{g} {s['DCGM_FI_DEV_SM_CLOCK']}MHz {s['DCGM_FI_DEV_GPU_TEMP']}C {s['DCGM_FI_DEV_POWER_USAGE']:.0f}/{s['DCGM_FI_DEV_POWER_MGMT_LIMIT']:.0f}W "
                f"util{s['DCGM_FI_DEV_GPU_UTIL']}% pcie x{s['DCGM_FI_DEV_PCIE_LINK_WIDTH']}" + (f" THROTTLE {'|'.join(decode_throttle(thr))}" if thr else ""))

    def t_rank_timings(self, k=8):
        c, t = self.c, self.c.t
        rt = c.rank_timings(t)
        k = max(1, min(int(k), 16))
        if rt["state"] != "running":
            if "hang" in rt:
                h = rt["hang"]
                return (f"job {rt['state']}: {h['ranks_waiting']} ranks blocked in {h['stuck_collective']} since {h['since']}Z; "
                        f"rank {h['rank_without_heartbeat']} ({h['node']} gpu{h['gpu']}) has sent no heartbeat since {h['since']}Z"), \
                    {"highlight": [{"node": h["node"], "gpu": h["gpu"]}]}
            return f"job {rt['state']}: no step timings available", {}
        ranks = rt["ranks"]
        comp = sorted(ranks, key=lambda r: -r["compute_s"])
        med_c = statistics.median(r["compute_s"] for r in ranks)
        med_w = statistics.median(r["comm_wait_s"] for r in ranks)
        lines = [f"flight recorder, last ~50 steps: step {rt['step_s']:.2f}s (healthy {HEALTHY_STEP_S:.2f}s) = slowest compute {comp[0]['compute_s']:.2f}s "
                 f"+ exposed comm {rt['exposed_comm_s']:.2f}s (p50 {sim.COMM_S:.2f}s) + data wait {ranks[0]['data_wait_s']:.2f}s + overhead {sim.OVERHEAD_S:.2f}s",
                 f"median rank: compute {med_c:.2f}s, comm wait {med_w:.2f}s",
                 f"slowest {k} ranks by compute:"]
        for r in comp[:k]:
            lines.append(f"  rank{r['rank']} {r['node']} gpu{r['gpu']}: compute {r['compute_s']:.3f}s, comm wait {r['comm_wait_s']:.3f}s")
        lines.append(f"NCCL: {rt['ib_devices_per_node']} IB device(s) per node in use, median bus bandwidth {rt['median_busbw_gbps']} GB/s")
        for ln in rt["slowest_links"]:
            lines.append(f"  slow link: {ln['node']} {ln['nic']} busbw {ln['busbw_gbps']} GB/s, retries/min {ln['retries_per_min']}")
        hl = [{"node": r["node"], "gpu": r["gpu"]} for r in comp[:8] if r["compute_s"] > med_c * 1.08]
        return "\n".join(lines), {"highlight": hl}

    def t_explain_slowdown(self, remove):
        if isinstance(remove, dict):
            remove = [remove]
        res = self.c.explain(self.c.t, remove)
        if "error" in res:
            return res["error"], {}
        return (f"current mean step {res['current_step_s']}s vs healthy {res['healthy_step_s']}s; "
                f"without {json.dumps(remove)}: ~{res['estimated_step_s_without']}s -> explains {res['explained_pct']}% of the slowdown"
                f" ({'; '.join(res['notes'])})"), {"explained_pct": res["explained_pct"]}

    def t_search_logs(self, pattern, node=None, minutes=60, limit=15):
        c, t = self.c, self.c.t
        minutes = max(1, min(int(minutes), 720))
        hits, total = c.search_logs(pattern, node, t - minutes * 60, t + 1e-6, limit=max(1, min(int(limit), 40)))
        if not hits:
            return f"no log lines matching /{pattern}/ in the last {minutes} min" + (f" on {node}" if node else ""), {}
        lines = [f"{total} matching lines in last {minutes} min (showing last {len(hits)}):"]
        for tt, n, src, lvl, msg in hits:
            lines.append(f"  {clock_str(tt)}Z {n} {src} {lvl}: {msg}")
        return "\n".join(lines), {}

    def t_change_log(self, minutes=180):
        c, t = self.c, self.c.t
        ev = [e for e in c.events if t - int(minutes) * 60 <= e["t"] <= t]
        if not ev:
            return f"no changes in the last {minutes} min", {}
        return "\n".join(f"{clock_str(e['t'])}Z {e['actor']} {e['action']} {e['target']}: {e['detail']}" for e in ev), {}

    def t_network_status(self):
        c, t = self.c, self.c.t
        ver, cfg = c.config_at(t)
        rows = []
        for node in c.members(t).values():
            for h in range(GPUS_PER_NODE):
                a, b = c.ib_sample(node, h, t), c.ib_sample(node, h, t - 600)
                d_sym = a["symbol_error"] - b["symbol_error"]
                d_ld = a["link_downed"] - b["link_downed"]
                rows.append((d_sym + 1000 * d_ld, node, h, a, d_sym, d_ld))
        rows.sort(key=lambda r: -r[0])
        rates = [r[3]["rate_gbps"] for r in rows]
        hcas = cfg["NCCL_IB_HCA"].split(",")
        lines = [f"IB fabric (64 nodes x 8 NICs, 400 Gb/s each). NCCL_IB_HCA in effect ({ver}): {cfg['NCCL_IB_HCA']} ({len(hcas)} device(s)/node)",
                 f"per-NIC tx rate: median {statistics.median(rates):.1f} Gb/s, max {max(rates):.1f} Gb/s"]
        noisy = [r for r in rows if r[0] > 0]
        if noisy:
            lines.append("NICs with new errors (last 10 min):")
            for score, node, h, a, d_sym, d_ld in noisy[:5]:
                lines.append(f"  {node} mlx5_{h}: {a['state']}, +{d_sym} symbol errors, +{d_ld} link downs, {a['rate_gbps']} Gb/s")
        else:
            lines.append("no new InfiniBand errors on any in-job NIC in the last 10 min")
        idle = sum(1 for r in rows if r[3]["rate_gbps"] < 1.0)
        if idle:
            lines.append(f"{idle} of {len(rows)} NICs carry ~no traffic")
        hl = [{"node": r[1]} for r in rows[:1] if r[0] > 100]
        return "\n".join(lines), {"highlight": hl}

    def t_storage_status(self):
        c, t = self.c, self.c.t
        s, before = c.storage_sample(t), c.storage_sample(t - 1800)
        dw = c.data_wait(t) if c.job_state(t) == "running" else 0.0
        lines = [f"{s['volume']}: read latency {s['read_latency_ms']} ms (30m ago {before['read_latency_ms']} ms) | local NVMe shard cache: {s['local_nvme_cache']}",
                 "top clients now: " + ", ".join(f"{x['job']} {x['read_gbps']} GB/s" for x in s["clients"]),
                 "30m ago: " + ", ".join(f"{x['job']} {x['read_gbps']} GB/s" for x in before["clients"]),
                 f"dataloader wait per step (flight recorder): {dw:.2f}s"]
        return "\n".join(lines), {}

    def t_job_config(self, version=None):
        c, t = self.c, self.c.t
        ver, cfg = c.config_at(t)
        lines = [f"in effect: {ver}"]
        show = c.configs.get(version or ver)
        if show is None:
            return f"unknown version {version}; known: {sorted(c.configs)}", {}
        lines += [f"  {k}={v}" for k, v in show.items()]
        vs = sorted(c.configs)
        for a, b in zip(vs, vs[1:]):
            diff = [f"{k}: {c.configs[a][k]!r} -> {c.configs[b][k]!r}" for k in c.configs[b] if c.configs[a].get(k) != c.configs[b][k]]
            lines.append(f"diff {a} -> {b}: " + ("; ".join(diff) or "none"))
        return "\n".join(lines), {}

    def t_collect_diagnostics(self, node):
        c, t = self.c, self.c.t
        if node not in c._nidx:
            return f"error: unknown node {node}", {}
        lines = [f"crusoe diagnostics vm create {node} -> nvidia_bug_report (completed {clock_str(t)}Z)"]
        for g in range(GPUS_PER_NODE):
            s = c.gpu_sample(node, g, t)
            if s is None:
                lines.append(f"GPU 00000000:{sim.GPU_BUS[g]}:00.0: Unable to determine the device handle: GPU is lost")
                continue
            thr = s["DCGM_FI_DEV_CLOCK_THROTTLE_REASONS"]
            lines.append(f"GPU 00000000:{sim.GPU_BUS[g]}:00.0: SM {s['DCGM_FI_DEV_SM_CLOCK']} MHz (max 1980) | temp {s['DCGM_FI_DEV_GPU_TEMP']} C "
                         f"(max operating 87, slowdown 89, shutdown 92) | power {s['DCGM_FI_DEV_POWER_USAGE']:.0f}/{s['DCGM_FI_DEV_POWER_MGMT_LIMIT']:.0f} W "
                         f"| clocks_event_reasons: {', '.join(decode_throttle(thr))} | PCIe Gen{s['DCGM_FI_DEV_PCIE_LINK_GEN']} x{s['DCGM_FI_DEV_PCIE_LINK_WIDTH']} (max x16) "
                         f"| ECC SBE {s['DCGM_FI_DEV_ECC_SBE_VOL_TOTAL']} DBE {s['DCGM_FI_DEV_ECC_DBE_VOL_TOTAL']}")
        xids, _ = c.search_logs(r"Xid|fallen off|PCIe bandwidth|AER", node, t - 86400, t + 1e-6, limit=6)
        lines.append("kernel log (XID / PCIe): " + (" || ".join(f"{clock_str(x[0])}Z {x[4]}" for x in xids) if xids else "none"))
        return "\n".join(lines), {"highlight": [{"node": node}]}

    # ---------------------------------------------------------------- actions
    def t_file_support_ticket(self, node, summary):
        res = self.c.action("file_support_ticket", {"node": node, "summary": summary})
        return f"filed {res['ticket']} for {node} with diagnostics attached", {}

    def act(self, name: str, args: dict, who: str = "agent") -> str:
        """Execute a remediation (approval is handled by the caller)."""
        args = dict(args or {})
        if name == "file_support_ticket":
            return self.run(name, args, who)[0]
        res = self.c.action(name, args, actor=who)
        self.calls.append({"t": self.c.t, "who": who, "name": name, "args": args})
        return json.dumps(res)

    def verify(self, steps: int = 200) -> dict:
        """Measure exactly `steps` training steps once the fix has taken effect.

        Waits for the job to be running, for any warm-up (e.g. the NVMe cache)
        and a 60 s settle, then judges throughput and new errors on nodes that
        are still in the job.
        """
        c = self.c
        steps = max(20, min(int(steps), 400))
        t_start = c.t
        deadline = c.t + 2700
        while c.job_state(c.t) != "running" and c.t < deadline:
            c.advance(15)
        settle_until = c.t + 60
        if c.cache_enabled_at is not None:
            settle_until = max(settle_until, c.cache_enabled_at + 30)
        c.advance_to(max(c.t, settle_until))
        n0 = len(c.steps)
        resume_t = c.t
        while len(c.steps) - n0 < steps and c.t < deadline:
            c.advance(20)
        window = c.steps[n0:n0 + steps]
        state = c.job_state(c.t)
        if len(window) < 5:
            return {"pass": False, "steps": len(window), "state": state, "ratio": 0.0,
                    "summary": f"job {state}: only {len(window)} steps completed in {(c.t - t_start) / 60:.0f} min"}
        t_first = max(resume_t, (c.steps[n0 - 1][0] if n0 > 0 else window[0][0] - window[0][2]))
        dur = window[-1][0] - t_first
        tps = len(window) * TOKENS_PER_STEP / dur if dur > 0 else 0.0
        ratio = tps / BASELINE_TPS
        p50 = statistics.median(s[2] for s in window)
        p95 = sorted(s[2] for s in window)[int(0.95 * len(window)) - 1]
        hits, _ = c.search_logs(r"Xid|NCCL WARN|Link down|Watchdog|prefetch queue empty", None, t_first, c.t + 1e-6, limit=40)
        errs = [e for e in hits if c.is_member(e[1], e[0])]
        ok = ratio >= 0.97 and not errs and state == "running"
        summary = (f"{len(window)} steps {clock_str(t_first)}Z-{clock_str(window[-1][0])}Z: {tps / 1000:.1f}k tok/s = {100 * ratio:.1f}% of baseline, "
                   f"step p50 {p50:.2f}s p95 {p95:.2f}s, new error lines on in-job nodes: {len(errs)}, loss trend normal -> {'PASS' if ok else 'FAIL'}")
        return {"pass": ok, "ratio": round(ratio, 4), "steps": len(window), "state": state,
                "errors": [e[4] for e in errs[:5]], "summary": summary}
