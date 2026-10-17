"""Simulated LLM pre-training run on a 512-GPU Crusoe H100 cluster.

The simulator is the ground truth. Agents only see what a real on-call
engineer would: DCGM metrics, InfiniBand counters, JournalD / NCCL / trainer
logs, the change log, Crusoe Instance Health and a PyTorch-style flight
recorder of per-rank timings. Faults, red herrings and remediation effects
are derived deterministically from the scenario seed, so any run replays
exactly.
"""
from __future__ import annotations

import math
import re
import zlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

GPUS_PER_NODE = 8
N_ACTIVE = 64
N_SPARE = 2
N_GPUS = N_ACTIVE * GPUS_PER_NODE
INSTANCE_TYPE = "h100-80gb-sxm-ib.8x"
LOCATION = "us-east1-a"
CLUSTER = "training-east"
NODEPOOL = "np-9addff51"
JOB = "yuzu-70b-pretrain"
EPOCH = datetime(2026, 10, 17, 1, 30, tzinfo=timezone.utc)

BASE_CLOCK = 1980.0          # MHz, H100 SXM max boost
IDLE_CLOCK = 345.0
TDP_W = 700.0
COMPUTE_S = 3.55             # per-step compute on a healthy GPU
COMM_S = 0.62                # exposed (non-overlapped) collective time
OVERHEAD_S = 0.23            # optimizer step, logging, misc
BASE_STEP_S = COMPUTE_S + COMM_S + OVERHEAD_S
TOKENS_PER_STEP = 512 * 4096
CKPT_EVERY = 100
CKPT_SAVE_S = 4.0
SLOW_CKPT_SAVE_S = 140.0
HEALTHY_STEP_S = BASE_STEP_S + CKPT_SAVE_S / CKPT_EVERY
BASELINE_TPS = TOKENS_PER_STEP / HEALTHY_STEP_S
START_STEP = 18_240
SCRAPE_S = 15.0
SWAP_NODE_S = 240.0          # cordon + drain + attach a hot spare
RESTART_S = 210.0            # relaunch 512 ranks, NCCL init, load checkpoint
WATCHDOG_S = 1800.0          # NCCL watchdog timeout configured for the job
CACHE_WARMUP_S = 180.0
GPU_BUS = ["18", "2A", "3A", "5D", "9A", "AB", "BA", "DB"]
HCA_BUS = ["19", "2B", "3B", "5E", "9B", "AC", "BB", "DC"]
ALL_HCAS = ",".join(f"mlx5_{i}" for i in range(GPUS_PER_NODE))

THROTTLE_REASONS = [
    (0x01, "GpuIdle"), (0x02, "ApplicationsClocksSetting"), (0x04, "SwPowerCap"),
    (0x08, "HwSlowdown"), (0x10, "SyncBoost"), (0x20, "SwThermalSlowdown"),
    (0x40, "HwThermalSlowdown"), (0x80, "HwPowerBrakeSlowdown"),
]

CONFIG_V42 = {
    "NCCL_IB_HCA": ALL_HCAS,
    "NCCL_SOCKET_IFNAME": "eth0",
    "NCCL_NET_GDR_LEVEL": "PHB",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    "NCCL_TIMEOUT_MIN": "30",
    "CHECKPOINT_DIR": "/mnt/ckpt-fast/yuzu-70b",
    "DATA_DIR": "/mnt/datasets/yuzu-mix-v3",
    "PARALLELISM": "TP=8 PP=4 DP=16",
    "GLOBAL_BATCH": "512 x 4096 tokens",
}

# Which model-facing category each simulated fault belongs to.
CATEGORY = {
    "thermal": "gpu_thermal",
    "power_cap": "gpu_power_cap",
    "ib_flap": "ib_link",
    "nccl_regression": "nccl_config",
    "dataloader_starvation": "storage_contention",
    "xid79": "gpu_hardware_failure",
    "checkpoint_stall": "checkpoint_storage",
    "cpu_contention": "cpu_contention",
    "pcie_downgrade": "pcie_link",
}

_MASK = (1 << 64) - 1


def _h(*parts) -> int:
    """splitmix64-style hash; stable across processes (unlike hash())."""
    x = 0x9E3779B97F4A7C15
    for p in parts:
        if isinstance(p, str):
            p = zlib.crc32(p.encode())
        elif isinstance(p, float):
            p = int(p * 1000)
        elif p is None:
            p = 7
        x = (x ^ (p & _MASK)) & _MASK
        x = (x * 0xBF58476D1CE4E5B9) & _MASK
        x ^= x >> 27
        x = (x * 0x94D049BB133111EB) & _MASK
        x ^= x >> 31
    return x


def _u(*parts) -> float:
    return (_h(*parts) >> 11) / float(1 << 53)


def _n(*parts) -> float:
    u1 = max(_u(*parts, 1), 1e-12)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * _u(*parts, 2))


def decode_throttle(mask: int) -> list[str]:
    return [name for bit, name in THROTTLE_REASONS if mask & bit] or ["None"]


def clock_str(t: float) -> str:
    return (EPOCH + timedelta(seconds=t)).strftime("%H:%M:%S")


def iso(t: float) -> str:
    return (EPOCH + timedelta(seconds=t)).strftime("%Y-%m-%dT%H:%M:%SZ")


def node_name(i: int) -> str:
    return f"{NODEPOOL}-{i}"


def node_ip(i: int) -> str:
    return f"10.20.{i // 200}.{i % 200 + 10}"


@dataclass
class Fault:
    kind: str
    t0: float
    node: str | None = None
    gpu: int | None = None
    hca: int | None = None
    sev: float = 1.0
    t_end: float = math.inf      # when the fault stopped affecting the job
    fixed_by: str | None = None
    extra: dict | None = None

    def active(self, t: float) -> bool:
        return self.t0 <= t < self.t_end

    def ramp(self, t: float, seconds: float = 180.0) -> float:
        return 0.0 if t < self.t0 else min(1.0, (t - self.t0) / seconds)


class Cluster:
    """One training job on one CMK cluster, observed through tools."""

    def __init__(self, scenario: dict):
        self.scenario = scenario
        self.seed = int(scenario["seed"])
        self.nodes = [node_name(i) for i in range(1, N_ACTIVE + N_SPARE + 1)]
        self._nidx = {n: i for i, n in enumerate(self.nodes, start=1)}
        self.spares = list(self.nodes[N_ACTIVE:])
        # membership history: list of (t, slot, node); slot s initially node s+1
        self._member_changes: list[tuple[float, int, str]] = []
        self._slots0 = {s: self.nodes[s] for s in range(N_ACTIVE)}
        self.cordoned: dict[str, float] = {}
        self.faults = [Fault(**f) for f in scenario["faults"]]
        self.herrings = scenario.get("herrings", [])
        self.configs = {"v42": dict(CONFIG_V42)}
        self._config_changes: list[tuple[float, str]] = [(-1e9, "v42")]
        self.down: list[tuple[float, float, str]] = []
        self.steps: list[tuple[float, int, float]] = []   # (t_end, step, seconds)
        self.ckpts: list[tuple[float, int, float]] = []   # (t_end, step, save_s)
        self.rollbacks: list[tuple[float, int]] = []
        self.step = START_STEP
        self.last_ckpt = (START_STEP // CKPT_EVERY) * CKPT_EVERY
        self._cursor = 0.0
        self.t = 0.0
        self.events: list[dict] = []
        self.logs: list[tuple[float, str, str, str, str]] = []
        self.actions: list[dict] = []
        self.tickets: list[dict] = []
        self.cache_enabled_at: float | None = None
        self.reverted: dict[str, float] = {}
        self.paused_jobs: dict[str, float] = {}
        self.disruptions = 0
        self._mcache: dict = {}
        self._stcache: dict = {}
        self._degraded_spare = self.nodes[-1]
        self._install_scenario()

    # ------------------------------------------------------------ scenario
    def _install_scenario(self):
        s = self.seed
        # Background change log so the audit trail isn't suspiciously empty.
        self._event(-86400 * 2.1, "ml-platform", "job.launch", JOB,
                    "launched with launcher config v42 on 64x h100-80gb-sxm-ib.8x")
        self._event(-5400, "autoscaler", "nodepool.scale", "np-cpu-a2", "scaled 3 -> 4 nodes")
        self._log(-86400 * 2.1 + 95, self.nodes[0], "nccl", "INFO",
                  "NCCL INFO NET/IB : Using " + " ".join(
                      f"[{i}]mlx5_{i}:1/IB" for i in range(8)) + f" ; OOB eth0:{node_ip(1)}<0>")
        for f in self.faults:
            getattr(self, f"_install_{f.kind}")(f)
        for h in self.herrings:
            getattr(self, f"_herring_{h['kind']}")(h)
        self.logs.sort(key=lambda r: r[0])
        self.events.sort(key=lambda e: e["t"])
        del s

    def _event(self, t, actor, action, target, detail):
        self.events.append({"t": t, "actor": actor, "action": action,
                            "target": target, "detail": detail})

    def _log(self, t, node, source, level, msg):
        self.logs.append((t, node, source, level, msg))

    def _scheduled_restart(self, f: Fault, version: str, overrides: dict, pr: str):
        self.configs[version] = {**CONFIG_V42, **overrides}
        t_stop = f.t0 - RESTART_S
        self._event(t_stop - 600, "ci-bot", "config.publish", "launcher-config",
                    f"{version} published ({pr}); diff: " + ", ".join(
                        f"{k}: {CONFIG_V42[k]!r} -> {v!r}" for k, v in overrides.items()))
        self._event(t_stop, "scheduler", "job.requeue", JOB,
                    "requeued after scheduled maintenance window; relaunching with launcher config "
                    + version)
        self.down.append((t_stop, f.t0, "scheduled requeue"))
        self._config_changes.append((f.t0, version))
        self.rollbacks.append((t_stop, -1))   # resolved lazily in _run_until

    def _install_thermal(self, f):
        pass  # silent gray failure: only DCGM shows it

    def _install_power_cap(self, f):
        cap = (f.extra or {}).get("cap", 450)
        bus = GPU_BUS
        if f.gpu is None:
            self._event(f.t0 - 20, "argocd", "daemonset.rollout", "gpu-power-policy",
                        f"v2 rolled out to canary node {f.node} (efficiency mode: {cap} W)")
            gpus = range(GPUS_PER_NODE)
        else:
            self._event(f.t0 - 35, "jlee", "ssh.session", f.node,
                        "interactive session opened (ticket OPS-2291: power draw test)")
            gpus = [f.gpu]
        for g in gpus:
            self._log(f.t0, f.node, "nvidia-smi", "INFO",
                      f"Power limit for GPU 00000000:{bus[g]}:00.0 was set to {cap:.2f} W "
                      f"from {TDP_W:.2f} W.")

    def _install_ib_flap(self, f):
        pass  # periodic kernel + NCCL lines, see _periodic_logs

    def _install_nccl_regression(self, f):
        self._scheduled_restart(f, "v43", {"NCCL_IB_HCA": "mlx5_0"},
                                "PR #812 'pin NCCL_IB_HCA for new rail topology'")
        self._log(f.t0 + 40, self.nodes[0], "nccl", "INFO",
                  "NCCL INFO NCCL_IB_HCA set by environment to mlx5_0")
        self._log(f.t0 + 41, self.nodes[0], "nccl", "INFO",
                  f"NCCL INFO NET/IB : Using [0]mlx5_0:1/IB ; OOB eth0:{node_ip(1)}<0>")

    def _install_checkpoint_stall(self, f):
        self._scheduled_restart(f, "v43", {"CHECKPOINT_DIR": "/mnt/archive/yuzu-70b"},
                                "PR #815 'move checkpoints off the fast tier'")

    def _install_dataloader_starvation(self, f):
        self._event(f.t0 - 25, "research-evals", "job.launch", "eval-sweep-7",
                    "started on 4 nodes (nodepool np-l40s-e1) reading /mnt/datasets")

    def _install_xid79(self, f):
        b = GPU_BUS[f.gpu]
        self._log(f.t0, f.node, "kernel", "ERR",
                  f"NVRM: Xid (PCI:0000:{b}:00): 79, pid='<unknown>', name=<unknown>, "
                  "GPU has fallen off the bus.")
        self._log(f.t0 + 0.4, f.node, "kernel", "ERR",
                  f"NVRM: GPU 0000:{b}:00.0: GPU has fallen off the bus.")
        self._log(f.t0 + 0.9, f.node, "kernel", "ERR",
                  "NVRM: A GPU crash dump has been created. If possible, please run "
                  "NVIDIA bug report.")

    def _install_cpu_contention(self, f):
        self._event(f.t0 - 90, "jlee", "ssh.session", f.node,
                    "interactive session opened (debugging eval scoring)")

    def _install_pcie_downgrade(self, f):
        b, bridge = GPU_BUS[f.gpu], f"{int(GPU_BUS[f.gpu], 16) - 1:02X}"
        for k in range(3):
            self._log(f.t0 - 4 + k, f.node, "kernel", "WARNING",
                      f"pcieport 0000:{bridge}:00.0: AER: Corrected error message received "
                      f"from 0000:{b}:00.0")
        self._log(f.t0, f.node, "kernel", "INFO",
                  f"nvidia 0000:{b}:00.0: 252.048 Gb/s available PCIe bandwidth, limited by "
                  f"32.0 GT/s PCIe x8 link at 0000:{bridge}:00.0 (capable of 504.112 Gb/s with "
                  "32.0 GT/s PCIe x16 link)")

    def _herring_fluentbit(self, h):
        self._event(h["t"], "argocd", "daemonset.rollout", "fluent-bit",
                    "v7 rolled out to all nodes (log flush interval 5s -> 1s)")

    def _herring_old_xid13(self, h):
        b = GPU_BUS[h["gpu"]]
        self._log(-3 * 3600 - 412, h["node"], "kernel", "ERR",
                  f"NVRM: Xid (PCI:0000:{b}:00): 13, pid=29381, name=python3, Graphics SM Warp "
                  "Exception on (GPC 4, TPC 0, SM 1): Out Of Range Address")

    def _herring_warm_gpu(self, h):
        pass  # visible in DCGM only

    def _herring_ib_sbe(self, h):
        pass  # static symbol-error count, see ib_sample

    def _herring_spare_degraded(self, h):
        self._log(-5 * 3600, self.nodes[-1], "kernel", "ERR",
                  f"NVRM: Xid (PCI:0000:{GPU_BUS[2]}:00): 140, Unrecovered ECC Error")

    def _herring_eval_job(self, h):
        self._event(h["t"], "research-evals", "job.launch", "eval-mini-3",
                    "started on 1 node (nodepool np-l40s-e1), reading /mnt/eval-sets")

    # ------------------------------------------------------------ membership
    def _members_entry(self, t: float):
        n = 0
        for tc, _, _ in self._member_changes:
            if tc <= t:
                n += 1
        hit = self._mcache.get(n)
        if hit is None:
            slots = dict(self._slots0)
            for _, slot, node in self._member_changes[:n]:
                slots[slot] = node
            hit = (slots, set(slots.values()))
            self._mcache[n] = hit
        return hit

    def members(self, t: float) -> dict[int, str]:
        return self._members_entry(t)[0]

    def is_member(self, node: str, t: float) -> bool:
        return node in self._members_entry(t)[1]

    def slot_of(self, node: str, t: float) -> int | None:
        for s, n in self.members(t).items():
            if n == node:
                return s
        return None

    def rank_of(self, node: str, gpu: int, t: float) -> int | None:
        s = self.slot_of(node, t)
        return None if s is None else s * GPUS_PER_NODE + gpu

    def node_of_rank(self, rank: int, t: float) -> tuple[str, int]:
        return self.members(t)[rank // GPUS_PER_NODE], rank % GPUS_PER_NODE

    def config_at(self, t: float) -> tuple[str, dict]:
        ver = "v42"
        for tc, v in self._config_changes:
            if tc <= t:
                ver = v
        return ver, self.configs[ver]

    # ------------------------------------------------------------ state
    def _in_down(self, t: float):
        for a, b, why in self.down:
            if a <= t < b:
                return (a, b, why)
        return None

    def _next_down_start(self, t: float):
        starts = [a for a, b, _ in self.down if a > t]
        return min(starts) if starts else None

    def _dead_member_fault(self, t: float):
        for f in self.faults:
            if f.kind == "xid79" and f.active(t) and self.is_member(f.node, t):
                return f
        return None

    def job_state(self, t: float) -> str:
        if self._in_down(t):
            return "restarting"
        dead = self._dead_member_fault(t)
        if dead:
            relaunched = [a for a, b, _ in self.down if dead.t0 < b <= t]
            if relaunched:
                return "failed_launch"
            return "crashed" if t >= dead.t0 + WATCHDOG_S else "hung"
        return "running"

    def _faults_on(self, node: str, t: float):
        return [f for f in self.faults if f.node == node and f.active(t)]

    def _thermal_clock(self, f: Fault, t: float) -> float:
        r = f.ramp(t)
        return BASE_CLOCK - r * 585.0 * f.sev + 18.0 * _n(self.seed, "thc", int(t // SCRAPE_S))

    def _cap_clock(self, f: Fault) -> float:
        cap = (f.extra or {}).get("cap", 450)
        return 1560.0 + (cap - 450) * 1.5

    def rank_factor(self, node: str, gpu: int, t: float) -> float:
        fac = 1.0
        for f in self._faults_on(node, t):
            if f.kind == "thermal" and f.gpu == gpu:
                fac = max(fac, BASE_CLOCK / max(self._thermal_clock(f, t), 900.0))
            elif f.kind == "power_cap" and (f.gpu is None or f.gpu == gpu):
                fac = max(fac, BASE_CLOCK / self._cap_clock(f))
            elif f.kind == "cpu_contention":
                fac = max(fac, 1.0 + 0.24 * f.sev + 0.02 * _n(self.seed, "cpu", int(t // 30)))
        return fac

    def comm_factor(self, t: float) -> float:
        fac = 1.0
        for f in self.faults:
            if not f.active(t) or (f.node and not self.is_member(f.node, t)):
                continue
            if f.kind == "ib_flap":
                fac = max(fac, 1.0 + 1.9 * f.sev * (0.7 + 0.6 * _u(self.seed, "ibj", int(t // 20))))
            elif f.kind == "pcie_downgrade":
                fac = max(fac, 1.0 + 0.95 * f.sev)
        _, cfg = self.config_at(t)
        if cfg["NCCL_IB_HCA"] != ALL_HCAS:
            fac = max(fac, 3.4)
        return fac

    def data_wait(self, t: float) -> float:
        for f in self.faults:
            if f.kind == "dataloader_starvation" and f.active(t):
                return 1.25 * f.sev * (0.85 + 0.3 * _u(self.seed, "dw", int(t // 30)))
        return 0.0

    def ckpt_save_s(self, t: float) -> float:
        _, cfg = self.config_at(t)
        return SLOW_CKPT_SAVE_S if cfg["CHECKPOINT_DIR"].startswith("/mnt/archive") else CKPT_SAVE_S

    def step_time(self, t: float) -> float | None:
        """Seconds for a step starting at t, or None if the job can't progress."""
        key = (round(t, 2), len(self.actions), len(self.down))
        if key in self._stcache:
            return self._stcache[key]
        if self.job_state(t) != "running":
            val = None
        else:
            fmax = 1.0
            mem = self._members_entry(t)[1]
            for f in self.faults:
                if f.node in mem and f.active(t) and f.kind in ("thermal", "power_cap", "cpu_contention"):
                    for g in range(GPUS_PER_NODE):
                        fmax = max(fmax, self.rank_factor(f.node, g, t))
            st = COMPUTE_S * fmax + COMM_S * self.comm_factor(t) + OVERHEAD_S + self.data_wait(t)
            val = st * (1.0 + 0.006 * _n(self.seed, "st", int(t // 4)))
        if len(self._stcache) > 50_000:
            self._stcache.clear()
        self._stcache[key] = val
        return val

    def _run_until(self, T: float):
        t = self._cursor
        while t < T:
            dn = self._in_down(t)
            if dn:
                # scheduled (scenario) requeue: roll back to last checkpoint once
                for i, (tr, lost) in enumerate(self.rollbacks):
                    if lost == -1 and tr <= t:
                        self.rollbacks[i] = (tr, self.step - self.last_ckpt)
                        self.step = self.last_ckpt
                t = min(T, dn[1])
                continue
            st = self.step_time(t)
            if st is None:
                t = T
                break
            save = 0.0
            if (self.step + 1) % CKPT_EVERY == 0:
                save = self.ckpt_save_s(t)
            nxt = self._next_down_start(t)
            if nxt is not None and nxt < t + st + save:
                t = nxt          # the in-flight step is lost when the job stops
                continue
            if t + st + save > T:
                break
            t += st + save
            self.step += 1
            self.steps.append((t, self.step, st + save))
            if save:
                self.last_ckpt = self.step
                self.ckpts.append((t, self.step, save))
        self._cursor = max(self._cursor, t)

    def advance(self, dt: float):
        self.t += max(0.0, dt)
        self._run_until(self.t)

    def advance_to(self, t: float):
        self.advance(t - self.t)

    # ------------------------------------------------------------ telemetry
    def throughput(self, t_a: float, t_b: float) -> float | None:
        """Mean tokens/s over [t_a, t_b) including downtime and checkpoint saves."""
        span = t_b - t_a
        if span <= 0:
            return None
        done = [s for s in self.steps if t_a <= s[0] < t_b]
        net = len(done)
        for tr, lost in self.rollbacks:
            if t_a <= tr < t_b and lost > 0:
                net -= lost
        return max(0, net) * TOKENS_PER_STEP / span

    def recent_steps(self, t: float, window_s: float = 300.0):
        return [s for s in self.steps if t - window_s <= s[0] <= t]

    def series(self, t_a: float, t_b: float, bucket: float = 30.0):
        pts = []
        steps = [s for s in self.steps if t_a - 60 <= s[0] < t_b]
        k = 0
        t = t_a
        while t < t_b:
            n_tok = 0.0
            while k < len(steps) and steps[k][0] < t + bucket:
                if steps[k][0] >= t:
                    n_tok += TOKENS_PER_STEP
                k += 1
            pts.append((round(t, 1), round(n_tok / bucket)))
            t += bucket
        return pts

    def _xid79_dead(self, node: str, gpu: int, t: float) -> bool:
        return any(f.kind == "xid79" and f.node == node and f.gpu == gpu and t >= f.t0
                   for f in self.faults)

    def gpu_sample(self, node: str, gpu: int, t: float) -> dict | None:
        """One DCGM scrape for one GPU (None = DCGM can't read it)."""
        if self._xid79_dead(node, gpu, t):
            return None
        k = int(t // SCRAPE_S)
        ni = self._nidx[node]
        seed = self.seed

        def nz(tag):
            return _n(seed, tag, ni, gpu, k)

        base_temp = 57.0 + 11.0 * _u(seed, "tb", ni, gpu)
        base_pw = 640.0 + 40.0 * _u(seed, "pb", ni, gpu)
        member = self.is_member(node, t)
        state = self.job_state(t)
        throttle, limit, pcie_w, xid = 0, TDP_W, 16, 0
        sbe = int(3 * _u(seed, "sbe", ni, gpu))
        replay = int(4 * _u(seed, "rp", ni, gpu))
        fb = 0
        if not member or state in ("restarting", "crashed", "failed_launch"):
            clock = IDLE_CLOCK + 6 * nz("ic")
            util, sm, tensor = 0.0, 0.0, 0.0
            power = 68.0 + 3 * nz("ip")
            temp = 31.0 + 4 * _u(seed, "it", ni, gpu) + nz("itn")
            throttle = 0x01
        elif state == "hung":
            clock = BASE_CLOCK - 15 - 8 * abs(nz("c"))
            util, sm, tensor = 100.0, 0.03 + 0.01 * abs(nz("sm")), 0.0
            power = 176.0 + 8 * nz("p")
            temp = base_temp - 17 + nz("t")
            fb = 75_100 + int(300 * _u(seed, "fb", ni, gpu))
        else:
            st = self.step_time(t) or BASE_STEP_S
            fac = self.rank_factor(node, gpu, t)
            contended = any(f.kind == "cpu_contention" for f in self._faults_on(node, t))
            busy = min(1.0, COMPUTE_S * (1.0 if contended else fac) / st)
            util = 100.0 * (1.0 - self.data_wait(t) / st)
            if contended:
                util -= 13.0 + 2 * abs(nz("cu"))
            util = max(0.0, min(100.0, util - 0.6 * abs(nz("u"))))
            sm = 0.86 * busy + 0.04 + 0.01 * nz("sm")
            tensor = 0.63 * busy + 0.01 * nz("ta")
            clock = BASE_CLOCK - 12 * abs(nz("c"))
            power = base_pw * (0.78 + 0.22 * busy) + 6 * nz("p")
            temp = base_temp + 1.5 * nz("t")
            fb = 75_100 + int(300 * _u(seed, "fb", ni, gpu))
        for f in self.faults:
            if f.node != node or t < f.t0:
                continue
            if f.kind == "thermal" and f.gpu == gpu:
                r = f.ramp(t)
                hot = base_temp + r * (90.2 - base_temp) + 0.4 * nz("th")
                if member and state == "running":
                    clock = self._thermal_clock(f, t)
                    temp = hot
                    power = 690.0 - r * 170.0 + 5 * nz("tp")
                else:
                    temp = max(temp, 48.0 + 4 * r)
                if temp >= 87.0:
                    throttle |= 0x60
                elif temp >= 84.0:
                    throttle |= 0x20
            elif f.kind == "power_cap" and (f.gpu is None or f.gpu == gpu) and f.active(t):
                cap = (f.extra or {}).get("cap", 450)
                limit = float(cap)
                if member and state == "running":
                    clock = self._cap_clock(f) + 15 * nz("pc")
                    power = cap - 2.0 - 2.0 * abs(nz("pp"))
                    temp = base_temp - 8.0 + nz("pt")
                    throttle |= 0x04
            elif f.kind == "pcie_downgrade" and f.gpu == gpu:
                pcie_w = 8
                replay += int(42 * (t - f.t0) / 60.0)
        for h in self.herrings:
            if h["kind"] == "warm_gpu" and h["node"] == node and h["gpu"] == gpu and member:
                temp = min(temp + 13.5, 84.3)
            if h["kind"] == "old_xid13" and h["node"] == node and h["gpu"] == gpu:
                xid = 13
        for f in self.faults:
            if f.kind == "xid79" and f.node == node and f.gpu == gpu:
                xid = 79
        return {
            "DCGM_FI_DEV_SM_CLOCK": round(clock),
            "DCGM_FI_DEV_GPU_TEMP": round(temp),
            "DCGM_FI_DEV_MEMORY_TEMP": round(temp + 6 + nz("mt")),
            "DCGM_FI_DEV_POWER_USAGE": round(power, 1),
            "DCGM_FI_DEV_POWER_MGMT_LIMIT": limit,
            "DCGM_FI_DEV_GPU_UTIL": round(util),
            "DCGM_FI_PROF_SM_ACTIVE": round(max(0.0, sm), 3),
            "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE": round(max(0.0, tensor), 3),
            "DCGM_FI_DEV_CLOCK_THROTTLE_REASONS": throttle,
            "DCGM_FI_DEV_XID_ERRORS": xid,
            "DCGM_FI_DEV_ECC_SBE_VOL_TOTAL": sbe,
            "DCGM_FI_DEV_ECC_DBE_VOL_TOTAL": 0,
            "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL": 0,
            "DCGM_FI_DEV_PCIE_LINK_GEN": 5,
            "DCGM_FI_DEV_PCIE_LINK_WIDTH": pcie_w,
            "DCGM_FI_DEV_PCIE_REPLAY_COUNTER": replay,
            "DCGM_FI_DEV_FB_USED": fb,
        }

    def ib_sample(self, node: str, hca: int, t: float) -> dict:
        ni = self._nidx[node]
        seed = self.seed
        k = int(t // SCRAPE_S)
        running = self.is_member(node, t) and self.job_state(t) == "running"
        _, cfg = self.config_at(t)
        rate = 0.0
        if running:
            if cfg["NCCL_IB_HCA"] != ALL_HCAS:
                rate = 391.0 + 3 * _n(seed, "ibr", ni, hca, k) if hca == 0 else 0.3
            else:
                rate = 132.0 + 9 * _n(seed, "ibr", ni, hca, k)
        sym = lde = rcv = ler = 0
        state = "Active"
        for h in self.herrings:
            if h["kind"] == "ib_sbe" and h["node"] == node and h["hca"] == hca:
                sym += 12 + int(0.6 * max(0.0, t) / 60.0)
        for f in self.faults:
            if f.kind == "ib_flap" and f.node == node and f.hca == hca and t >= f.t0:
                dt = t - f.t0
                sym += int(260 * f.sev * dt / 60.0)
                rcv += int(35 * f.sev * dt / 60.0)
                lde += int(dt // 150) + 1
                ler += int(dt // 75) + 1
                if (dt % 150) < 6:
                    state = "Down"
                if running:
                    rate *= 0.45 + 0.25 * _u(seed, "ibf", k)
        return {"rate_gbps": round(max(rate, 0.0), 1), "state": state,
                "symbol_error": sym, "link_downed": lde, "port_rcv_errors": rcv,
                "link_error_recovery": ler, "port_xmit_discards": int(rcv * 0.2)}

    def node_sample(self, node: str, t: float) -> dict:
        ni = self._nidx[node]
        k = int(t // SCRAPE_S)
        running = self.is_member(node, t) and self.job_state(t) in ("running", "hung")
        cpu = 36.0 + 3 * _n(self.seed, "cpu", ni, k) if running else 3.0 + _u(self.seed, "ci", ni, k)
        procs = []
        if running:
            slot = self.slot_of(node, t)
            for g in range(GPUS_PER_NODE):
                procs.append({"pid": 41000 + g * 17 + ni, "user": "train",
                              "cpu_pct": round(380 + 40 * _n(self.seed, "pc", ni, g, k)),
                              "cmd": f"python train.py --rank {slot * 8 + g}"})
        procs += [{"pid": 2211, "user": "root", "cpu_pct": 41, "cmd": "dcgm-exporter"},
                  {"pid": 1874, "user": "root", "cpu_pct": 12, "cmd": "crusoe-watch-agent"},
                  {"pid": 1990, "user": "root", "cpu_pct": 9, "cmd": "fluent-bit"}]
        for f in self._faults_on(node, t):
            if f.kind == "cpu_contention":
                cpu = 97.0 + _u(self.seed, "cc", k)
                ex = f.extra or {}
                procs.append({"pid": ex.get("pid", 48213), "user": ex.get("user", "jlee"),
                              "cpu_pct": 15120,
                              "cmd": ex.get("cmd", "python3 /home/jlee/eval/score_samples.py "
                                                   "--workers 160")})
        procs.sort(key=lambda p: -p["cpu_pct"])
        return {"cpu_util_pct": round(min(cpu, 100.0), 1), "vcpus": 176,
                "mem_used_pct": round(61 + 2 * _u(self.seed, "mem", ni), 1) if running else 9.0,
                "top_processes": procs[:6]}

    def storage_sample(self, t: float) -> dict:
        k = int(t // SCRAPE_S)
        running = self.job_state(t) == "running"
        ours = 9.6 + 0.3 * _n(self.seed, "sd", k) if running else 0.1
        lat = 1.9 + 0.2 * abs(_n(self.seed, "sl", k))
        clients = [{"job": JOB, "read_gbps": round(ours, 2)}]
        for f in self.faults:
            if f.kind == "dataloader_starvation" and t >= f.t0:
                noisy_paused = "eval-sweep-7" in self.paused_jobs and t >= self.paused_jobs["eval-sweep-7"]
                if not noisy_paused:
                    clients.append({"job": "eval-sweep-7", "read_gbps": round(11.8 + 0.4 * _n(self.seed, "ev", k), 2)})
                    lat = 21.0 + 2 * abs(_n(self.seed, "sl2", k))
                if f.active(t) and running:
                    clients[0]["read_gbps"] = round(2.7 + 0.2 * _n(self.seed, "sd2", k), 2)
                elif self.cache_enabled_at is not None and t >= self.cache_enabled_at and running:
                    clients[0]["read_gbps"] = round(0.4 + 0.05 * _n(self.seed, "sd3", k), 2)
        for h in self.herrings:
            if h["kind"] == "eval_job" and t >= h["t"]:
                clients.append({"job": "eval-mini-3", "read_gbps": 0.6})
        clients.sort(key=lambda c: -c["read_gbps"])
        return {"volume": "datasets (Crusoe Shared Disk, /mnt/datasets)",
                "read_latency_ms": round(lat, 1), "clients": clients,
                "local_nvme_cache": "enabled" if self.cache_enabled_at is not None and t >= self.cache_enabled_at else "disabled"}

    def instance_health(self, node: str, t: float) -> tuple[str, list[str]]:
        codes = []
        for f in self.faults:
            if f.kind == "xid79" and f.node == node and t >= f.t0:
                codes += ["XID 79", "GPUFellOffTheBus"]
        if node == self._degraded_spare and any(h["kind"] == "spare_degraded" for h in self.herrings):
            return "Degraded", ["XID 140"]
        if codes:
            return "Unhealthy", codes
        return "Healthy", []

    # ------------------------------------------------------------ logs
    def _periodic_logs(self, t_a: float, t_b: float):
        out = []
        for f in self.faults:
            if f.kind == "ib_flap" and t_b > f.t0:
                bus = HCA_BUS[f.hca]
                start = max(t_a, f.t0)
                n0 = int((start - f.t0) // 150)
                tt = f.t0 + n0 * 150
                while tt < t_b:
                    if tt >= t_a:
                        out.append((tt, f.node, "kernel", "WARNING",
                                    f"mlx5_core 0000:{bus}:00.0 ibp{int(bus, 16)}s0: Link down"))
                    if t_a <= tt + 6 < t_b:
                        out.append((tt + 6, f.node, "kernel", "INFO",
                                    f"mlx5_core 0000:{bus}:00.0 ibp{int(bus, 16)}s0: Link up"))
                    tt += 150
                if self.is_member(f.node, t_a) or self.is_member(f.node, t_b):
                    tt = f.t0 + 20 + 45 * int(max(0, start - f.t0) // 45)
                    ni = self._nidx[f.node]
                    while tt < min(t_b, f.t_end):
                        if tt >= t_a and self.job_state(tt) == "running":
                            peer = node_ip((ni + 1 + int(_u(self.seed, "peer", tt) * 60)) % 64 + 1)
                            out.append((tt, f.node, "nccl", "WARN",
                                        f"NCCL WARN NET/IB : Got completion from peer {peer}<52011> with "
                                        f"error 12, opcode 0, len 0, vendor err 129 (Send) on mlx5_{f.hca}"))
                        tt += 45
            if f.kind == "dataloader_starvation" and t_b > f.t0:
                tt = f.t0 + 30 + 60 * int(max(0, t_a - f.t0) // 60)
                while tt < min(t_b, f.t_end):
                    if tt >= t_a and self.job_state(tt) == "running":
                        w = self.data_wait(tt)
                        out.append((tt, self.members(tt)[0], "trainer", "WARNING",
                                    f"[data] rank 0: prefetch queue empty, waited {w:.2f}s for next "
                                    "batch (num_workers=8, prefetch_factor=2)"))
                    tt += 60
            if f.kind == "xid79" and t_a <= f.t0 + WATCHDOG_S < t_b and self._dead_member_fault(f.t0 + WATCHDOG_S):
                tw = f.t0 + WATCHDOG_S
                out.append((tw, self.nodes[0], "trainer", "ERR",
                            "[rank0]:[E ProcessGroupNCCL.cpp:616] [Rank 0] Watchdog caught collective "
                            "operation timeout: WorkNCCL(SeqNum=184211, OpType=ALLREDUCE, "
                            "Timeout(ms)=1800000) ran for 1800042 milliseconds before timing out."))
                out.append((tw + 2, self.nodes[0], "trainer", "ERR",
                            "torch.distributed.elastic.multiprocessing.errors.ChildFailedError: "
                            "train.py FAILED"))
        # trainer progress + checkpoint lines from the step record
        for (te, step, sec) in self.steps:
            if t_a <= te < t_b:
                if step % 20 == 0:
                    tps = TOKENS_PER_STEP / sec
                    out.append((te, self.members(te)[0], "trainer", "INFO",
                                f"step {step} | loss {self.loss(step):.4f} | {tps / 1000:.1f}k tok/s "
                                f"| step_time {sec:.2f}s"))
        for (te, step, save) in self.ckpts:
            if t_a <= te < t_b:
                _, cfg = self.config_at(te)
                out.append((te, self.members(te)[0], "trainer", "INFO",
                            f"Saving checkpoint to {cfg['CHECKPOINT_DIR']}/step_{step} ... done in "
                            f"{save:.1f}s"))
        return out

    def search_logs(self, pattern: str, node: str | None, t_a: float, t_b: float,
                    limit: int = 20) -> tuple[list, int]:
        rows = [r for r in self.logs if t_a <= r[0] < t_b] + self._periodic_logs(t_a, t_b)
        rx = re.compile(pattern, re.I) if pattern else None
        hits = [r for r in rows if (node is None or r[1] == node)
                and (rx is None or rx.search(r[4]) or rx.search(r[2]))]
        hits.sort(key=lambda r: r[0])
        return hits[-limit:], len(hits)

    def loss(self, step: int) -> float:
        return 1.70 + 1.9 * (1 + step / 3000.0) ** -0.6 + 0.012 * _n(self.seed, "loss", step)

    # ------------------------------------------------------------ flight recorder
    def rank_timings(self, t: float) -> dict:
        state = self.job_state(t)
        out = {"state": state, "ranks": []}
        if state != "running":
            if state in ("hung", "crashed"):
                dead = self._dead_member_fault(t)
                r = self.rank_of(dead.node, dead.gpu, t)
                out["hang"] = {"stuck_collective": "ALLREDUCE SeqNum=184211",
                               "since": clock_str(dead.t0),
                               "ranks_waiting": N_GPUS - 1,
                               "rank_without_heartbeat": r,
                               "node": dead.node, "gpu": dead.gpu}
            return out
        st = self.step_time(t) or BASE_STEP_S
        data = self.data_wait(t)
        for s, node in self.members(t).items():
            for g in range(GPUS_PER_NODE):
                c = COMPUTE_S * self.rank_factor(node, g, t) * (1 + 0.004 * _n(self.seed, "rc", s, g, int(t // 60)))
                out["ranks"].append({"rank": s * 8 + g, "node": node, "gpu": g,
                                     "compute_s": round(c, 3), "data_wait_s": round(data, 3),
                                     "comm_wait_s": round(max(0.0, st - c - data - OVERHEAD_S), 3)})
        out["step_s"] = round(st, 3)
        out["exposed_comm_s"] = round(COMM_S * self.comm_factor(t), 3)
        # per-link view (which NIC is slowest in the collective)
        links = []
        _, cfg = self.config_at(t)
        for f in self.faults:
            if f.active(t) and f.kind == "ib_flap" and self.is_member(f.node, t):
                links.append({"node": f.node, "nic": f"mlx5_{f.hca}", "busbw_gbps": round(55 + 20 * _u(self.seed, "bw", int(t // 20)), 1),
                              "retries_per_min": 140})
            if f.active(t) and f.kind == "pcie_downgrade" and self.is_member(f.node, t):
                links.append({"node": f.node, "nic": f"mlx5_{f.gpu}", "busbw_gbps": 94.0, "retries_per_min": 0})
        out["slowest_links"] = links
        out["ib_devices_per_node"] = len(cfg["NCCL_IB_HCA"].split(","))
        out["median_busbw_gbps"] = round(186.0 / (1.0 if cfg["NCCL_IB_HCA"] == ALL_HCAS else 5.6), 1)
        return out

    def explain(self, t: float, removes: list[dict]) -> dict:
        """Counterfactual: share of the slowdown that disappears if effects are removed.

        Uses only observable per-rank timings (critical path = slowest rank's
        compute + exposed comm + data wait + overhead) plus the checkpoint log,
        never the hidden fault list.
        """
        rt = self.rank_timings(t)
        if rt["state"] != "running":
            return {"error": f"job is {rt['state']}; no steps to decompose"}
        ranks = rt["ranks"]
        comp = {r["rank"]: r["compute_s"] for r in ranks}
        med_c = sorted(comp.values())[len(comp) // 2]
        data = ranks[0]["data_wait_s"]
        comm = rt["exposed_comm_s"]
        cur_step = max(comp.values()) + comm + OVERHEAD_S + data
        window = [s for s in self.steps if t - 900 <= s[0] <= t]
        ck = [c for c in self.ckpts if t - 900 <= c[0] <= t]
        ckpt_per_step = (sum(c[2] for c in ck) / len(window)) if window else CKPT_SAVE_S / CKPT_EVERY
        ckpt_per_step = max(ckpt_per_step, CKPT_SAVE_S / CKPT_EVERY)
        current = cur_step + ckpt_per_step
        notes = []
        new_ckpt = ckpt_per_step
        for rm in removes:
            kind = rm.get("kind")
            if kind in ("ranks", "gpu", "node"):
                targets = set(int(r) for r in (rm.get("ranks") or []))
                if kind == "gpu" and rm.get("node") is not None and rm.get("gpu") is not None:
                    r = self.rank_of(rm["node"], int(rm["gpu"]), t)
                    if r is not None:
                        targets.add(r)
                if kind == "node" and rm.get("node"):
                    sl = self.slot_of(rm["node"], t)
                    if sl is not None:
                        targets |= {sl * 8 + g for g in range(8)}
                hit = [r for r in targets if r in comp]
                for r in hit:
                    comp[r] = med_c
                notes.append(f"{len(hit)} rank(s) set to median compute ({med_c:.2f}s)")
            elif kind == "comm":
                comm = COMM_S
                notes.append(f"exposed comm set to its 7-day p50 ({COMM_S:.2f}s)")
            elif kind == "data_wait":
                data = 0.0
                notes.append("dataloader wait set to 0")
            elif kind == "checkpoint":
                new_ckpt = CKPT_SAVE_S / CKPT_EVERY
                notes.append(f"checkpoint saves set to their 7-day p50 ({CKPT_SAVE_S:.0f}s)")
            else:
                notes.append(f"ignored unknown removal {rm}")
        new = max(comp.values()) + comm + OVERHEAD_S + data + new_ckpt
        slowdown = current - HEALTHY_STEP_S
        pct = 0.0 if slowdown <= 0.03 else max(0.0, min(100.0, 100.0 * (current - new) / slowdown))
        return {"current_step_s": round(current, 3), "healthy_step_s": round(HEALTHY_STEP_S, 3),
                "estimated_step_s_without": round(new, 3), "explained_pct": round(pct, 1),
                "notes": notes}

    # ------------------------------------------------------------ actions
    def _restart(self, reason: str, exclude=(), config: str | None = None, actor: str = "agent"):
        now = self.t
        self._run_until(now)
        swap = SWAP_NODE_S if exclude else 0.0
        end = now + swap + RESTART_S
        self.down.append((now, end, reason))
        lost = self.step - self.last_ckpt
        self.rollbacks.append((now, lost))
        self.step = self.last_ckpt
        replaced = []
        for node in exclude:
            slot = self.slot_of(node, now)
            if slot is None:
                continue
            self.cordoned[node] = now
            for f in self.faults:
                if f.node == node and f.t_end > now:
                    f.t_end = now
                    f.fixed_by = f.fixed_by or reason
            if self.spares:
                spare = self.spares.pop(0)
                self._member_changes.append((now, slot, spare))
                replaced.append((node, spare))
            else:
                replaced.append((node, None))
        if config:
            self._config_changes.append((end, config))
            cfg = self.configs[config]
            for f in self.faults:
                if f.kind == "nccl_regression" and cfg["NCCL_IB_HCA"] == ALL_HCAS and f.t_end > end:
                    f.t_end, f.fixed_by = end, reason
                if f.kind == "checkpoint_stall" and not cfg["CHECKPOINT_DIR"].startswith("/mnt/archive") and f.t_end > end:
                    f.t_end, f.fixed_by = end, reason
            self._event(now, actor, "config.apply", "launcher-config", f"next launch uses {config}")
        self._event(now, actor, "job.restart", JOB,
                    f"{reason}; rolled back {lost} steps to checkpoint step_{self.last_ckpt}"
                    + "".join(f"; {a} replaced by {b or 'NO SPARE'}" for a, b in replaced))
        ver = config or self.config_at(now)[0]
        hcas = self.configs[ver]["NCCL_IB_HCA"].split(",")
        self._log(end - 15, self.members(end)[0], "nccl", "INFO",
                  "NCCL INFO NET/IB : Using " + " ".join(f"[{i}]{h}:1/IB" for i, h in enumerate(hcas))
                  + f" ; OOB eth0:{node_ip(self._nidx[self.members(end)[0]])}<0>")
        self._cursor = end
        self.disruptions += 1
        return {"downtime_s": swap + RESTART_S, "rolled_back_steps": lost,
                "resumes_at": clock_str(end), "replaced": replaced}

    def action(self, name: str, args: dict, actor: str = "agent") -> dict:
        """Apply a remediation. Returns an observable result."""
        before = [f.t_end for f in self.faults]
        rec = {"t": self.t, "name": name, "args": dict(args or {}), "actor": actor}
        self.actions.append(rec)
        res = self._action(name, dict(args or {}), actor)
        rec["fixed"] = [f.kind for f, b in zip(self.faults, before) if f.t_end != b]
        rec["ok"] = bool(res.get("ok"))
        return res

    def _action(self, name: str, args: dict, actor: str) -> dict:
        if name == "drain_node":
            node = args["node"]
            if not self.is_member(node, self.t):
                return {"ok": False, "error": f"{node} is not part of the job"}
            res = self._restart(f"drain {node} ({args.get('reason', 'requested')})", exclude=[node], actor=actor)
            return {"ok": True, **res}
        if name == "restart_job":
            excl = [n for n in (args.get("exclude_nodes") or []) if self.is_member(n, self.t)]
            cfg = args.get("config_version")
            if cfg and cfg not in self.configs:
                return {"ok": False, "error": f"unknown config {cfg}; known: {sorted(self.configs)}"}
            res = self._restart("restart requested", exclude=excl, config=cfg, actor=actor)
            return {"ok": True, **res}
        if name == "reset_power_limit":
            node, gpu = args["node"], args.get("gpu")
            prev = TDP_W
            for f in list(self.faults):
                if f.kind == "power_cap" and f.node == node and f.active(self.t) and (gpu is None or f.gpu in (None, gpu)):
                    f.t_end, f.fixed_by = self.t + 10, "reset_power_limit"
                    prev = float((f.extra or {}).get("cap", 450))
                    re = (f.extra or {}).get("reapply_s")
                    if re and "gpu-power-policy" not in self.reverted and f.gpu is None:
                        t_re = self.t + re
                        self.faults.append(Fault(kind="power_cap", t0=t_re, node=node, gpu=None, sev=f.sev,
                                                 extra={**(f.extra or {}), "reapplied": True}))
                        self._event(t_re - 2, "gpu-power-policy", "daemonset.reconcile", node,
                                    "drift detected: re-applied efficiency mode (450 W)")
                        for g in range(GPUS_PER_NODE):
                            self._log(t_re, node, "nvidia-smi", "INFO",
                                      f"Power limit for GPU 00000000:{GPU_BUS[g]}:00.0 was set to {prev:.2f} W from {TDP_W:.2f} W.")
                        self.logs.sort(key=lambda r: r[0])
                        self.events.sort(key=lambda e: e["t"])
            gpus = range(8) if gpu is None else [int(gpu)]
            for g in gpus:
                self._log(self.t + 5, node, "nvidia-smi", "INFO",
                          f"Power limit for GPU 00000000:{GPU_BUS[g]}:00.0 was set to {TDP_W:.2f} W from "
                          f"{prev:.2f} W.")
            self.advance(10)
            return {"ok": True, "power_limit_w": TDP_W, "gpus": list(gpus)}
        if name == "revert_rollout":
            target = str(args.get("rollout") or args.get("target") or args.get("name") or args.get("daemonset") or "")
            if "gpu-power-policy" in target:
                target = "gpu-power-policy"
            elif "fluent-bit" in target:
                target = "fluent-bit"
            self.reverted[target] = self.t
            if target == "gpu-power-policy":
                for f in self.faults:
                    if f.kind == "power_cap" and f.t0 > self.t and f.t_end > self.t:
                        f.t_end, f.fixed_by = f.t0, "revert_rollout"   # scheduled re-application never happens
                        self.events = [e for e in self.events if not (e["action"] == "daemonset.reconcile" and e["t"] > self.t)]
                        self.logs = [r for r in self.logs if not (r[2] == "nvidia-smi" and r[0] > self.t and "set to 450" in r[4])]
            self._event(self.t, actor, "rollout.revert", target, "rolled back to the previous revision")
            return {"ok": True, "reverted": target,
                    "note": "already-applied node state (e.g. power limits) is not changed by a rollback"}
        if name == "enable_local_cache":
            if self.cache_enabled_at is None:
                self.cache_enabled_at = self.t + CACHE_WARMUP_S
                for f in self.faults:
                    if f.kind == "dataloader_starvation" and f.t_end > self.cache_enabled_at:
                        f.t_end, f.fixed_by = self.cache_enabled_at, "enable_local_cache"
                self._event(self.t, actor, "data.cache", JOB, "local NVMe shard cache enabled (prefetch 64 shards)")
            return {"ok": True, "warm_at": clock_str(self.cache_enabled_at)}
        if name == "pause_job":
            job = args.get("job", "")
            self.paused_jobs[job] = self.t + 30
            for f in self.faults:
                if f.kind == "dataloader_starvation" and job == "eval-sweep-7" and f.t_end > self.t + 30:
                    f.t_end, f.fixed_by = self.t + 30, "pause_job"
            self._event(self.t, actor, "job.pause", job, "paused by on-call request")
            self.disruptions += 1
            return {"ok": True, "paused": job}
        if name == "kill_process":
            node, pid = args["node"], int(args["pid"])
            for f in self.faults:
                if f.kind == "cpu_contention" and f.node == node and (f.extra or {}).get("pid", 48213) == pid and f.active(self.t):
                    f.t_end, f.fixed_by = self.t + 5, "kill_process"
                    self._event(self.t, actor, "process.kill", node, f"killed pid {pid}")
                    self.advance(5)
                    return {"ok": True, "killed": pid}
            return {"ok": False, "error": f"pid {pid} not found on {node}"}
        if name == "file_support_ticket":
            self.tickets.append({"t": self.t, **args})
            return {"ok": True, "ticket": f"CRUSOE-{40000 + len(self.tickets) * 7 + self.seed % 100}"}
        return {"ok": False, "error": f"unknown action {name}"}

    # ------------------------------------------------------------ evaluation
    def incident_start(self) -> float:
        t0 = min(f.t0 for f in self.faults)
        if any(f.kind in ("nccl_regression", "checkpoint_stall") for f in self.faults):
            t0 -= RESTART_S
        return t0

    def alert_time(self) -> float:
        """When the throughput / progress alert fires (first 5-min window < 90%)."""
        start = self.incident_start()
        t = start + 60
        probe = Cluster(self.scenario)
        while t < start + 3600:
            probe.advance_to(t)
            if probe.job_state(t) in ("hung", "crashed"):
                last = probe.steps[-1][0] if probe.steps else 0
                if t - last >= 180:
                    return t
            elif t - 300 >= 0:
                # Alerts are suppressed while the job is (re)starting, so only
                # windows that begin after the latest relaunch count.
                resumed = max([b for a, b, _ in probe.down if b <= t], default=-1e9)
                if t - 300 >= resumed and probe.job_state(t) == "running":
                    tp = probe.throughput(t - 300, t)
                    if tp is not None and tp < 0.9 * BASELINE_TPS:
                        return t
            t += 15
        return start + 600

    def gpu_hours_lost(self) -> float:
        start = self.incident_start()
        before = [s for s in self.steps if s[0] <= start]
        step0 = before[-1][1] if before else START_STEP
        elapsed = self.t - start
        expected = elapsed / HEALTHY_STEP_S
        actual = self.step - step0
        return max(0.0, (expected - actual) * HEALTHY_STEP_S * N_GPUS / 3600.0)

    def ground_truth(self) -> list[dict]:
        out = []
        for f in self.faults:
            if (f.extra or {}).get("reapplied"):
                continue
            g = {"type": CATEGORY[f.kind], "kind": f.kind}
            if f.node:
                g["node"] = f.node
            if f.gpu is not None:
                g["gpu"] = f.gpu
            if f.hca is not None:
                g["nic"] = f"mlx5_{f.hca}"
            out.append(g)
        return out

    def grid_codes(self, t: float) -> str:
        """Per-GPU status for the UI: 64 job slots then spares/cordoned nodes.

        0 ok, 1 warm, 2 throttled/slow, 3 dead, 4 cordoned, 5 idle spare, 6 restarting.
        """
        codes = []
        state = self.job_state(t)
        order = list(self.members(t).values()) + [n for n in self.nodes if not self.is_member(n, t)]
        for node in order:
            for g in range(GPUS_PER_NODE):
                if self._xid79_dead(node, g, t):
                    codes.append("3")
                    continue
                if node in self.cordoned and self.cordoned[node] <= t:
                    codes.append("4")
                    continue
                if not self.is_member(node, t):
                    codes.append("5")
                    continue
                if state == "restarting":
                    codes.append("6")
                    continue
                smp = self.gpu_sample(node, g, t)
                thr = smp["DCGM_FI_DEV_CLOCK_THROTTLE_REASONS"] & 0xEC
                if thr or (state == "running" and smp["DCGM_FI_DEV_SM_CLOCK"] < 1800) or self.rank_factor(node, g, t) > 1.1:
                    codes.append("2")
                elif smp["DCGM_FI_DEV_GPU_TEMP"] >= 80:
                    codes.append("1")
                else:
                    codes.append("0")
        return "".join(codes), order
