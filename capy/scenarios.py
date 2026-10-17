"""Incident scenarios.

Dev scenarios are what we iterate prompts on. The test set is run once for
the scoreboard; it adds new seeds, three fault types that never appear in
dev or in any prompt (held-out), and two-fault combinations.
"""
from __future__ import annotations

from .sim import GPUS_PER_NODE, N_ACTIVE, _u, node_name

SEEN = ["thermal", "power_cap", "ib_flap", "nccl_regression", "dataloader_starvation", "xid79"]
HELDOUT = ["checkpoint_stall", "cpu_contention", "pcie_downgrade"]
COMBOS = [
    ("power_cap", "dataloader_starvation"),
    ("thermal", "nccl_regression"),
    ("ib_flap", "power_cap"),
    ("cpu_contention", "thermal"),
]
HERRINGS = ["fluentbit", "old_xid13", "warm_gpu", "ib_sbe", "spare_degraded", "eval_job"]


def _node(seed: int, tag, avoid: set) -> str:
    i = 1 + int(_u(seed, "node", tag) * N_ACTIVE)
    while node_name(i) in avoid:
        i = i % N_ACTIVE + 1
    return node_name(i)


def make(kinds, seed: int, split: str, sid: str | None = None) -> dict:
    kinds = list(kinds)
    used: set[str] = set()
    t_on = 1800.0 + int(_u(seed, "onset") * 1500)
    faults = []
    for j, kind in enumerate(kinds):
        t0 = t_on + j * (60 + int(_u(seed, "stagger", j) * 90))
        node = _node(seed, j, used)
        used.add(node)
        gpu = int(_u(seed, "gpu", j) * GPUS_PER_NODE)
        sev = round(0.9 + 0.25 * _u(seed, "sev", j), 3)
        f = {"kind": kind, "t0": float(t0), "sev": sev}
        if kind in ("thermal", "xid79", "pcie_downgrade"):
            f.update(node=node, gpu=gpu)
        elif kind == "power_cap":
            if _u(seed, "pcg", j) < 0.5:
                f.update(node=node, gpu=gpu, extra={"cap": 450})                     # manual leftover
            else:
                f.update(node=node, gpu=None, extra={"cap": 450, "reapply_s": 480})  # daemonset canary
        elif kind == "ib_flap":
            f.update(node=node, hca=int(_u(seed, "hca", j) * GPUS_PER_NODE))
        elif kind == "cpu_contention":
            f.update(node=node, extra={"pid": 48000 + int(_u(seed, "pid") * 900), "user": "jlee",
                                       "cmd": "python3 /home/jlee/eval/score_samples.py --workers 160"})
        faults.append(f)
    herrings = []
    picks = sorted(HERRINGS, key=lambda h: _u(seed, "herring", h))[:2]
    if "fluentbit" not in picks and _u(seed, "fb") < 0.5:
        picks.append("fluentbit")
    for h in picks:
        if h == "fluentbit":
            herrings.append({"kind": h, "t": t_on - 300 - int(_u(seed, "fbt") * 400)})
        elif h in ("old_xid13", "warm_gpu"):
            n = _node(seed, h, used)
            used.add(n)
            herrings.append({"kind": h, "node": n, "gpu": int(_u(seed, h, "g") * GPUS_PER_NODE)})
        elif h == "ib_sbe":
            n = _node(seed, h, used)
            used.add(n)
            herrings.append({"kind": h, "node": n, "hca": int(_u(seed, h, "h") * GPUS_PER_NODE)})
        elif h == "spare_degraded":
            herrings.append({"kind": h})
        elif h == "eval_job":
            herrings.append({"kind": h, "t": t_on - 600})
    return {"id": sid or f"{'+'.join(kinds)}-{seed}", "seed": seed, "kinds": kinds,
            "split": split, "heldout": any(k in HELDOUT for k in kinds),
            "faults": faults, "herrings": herrings}


def dev_set() -> list[dict]:
    return [make([k], s, "dev") for k in SEEN for s in (101, 102, 103)]


def test_set() -> list[dict]:
    out = [make([k], s, "test") for k in SEEN for s in (201, 202, 203, 204)]
    out += [make([k], s, "test") for k in HELDOUT for s in (301, 302, 303, 304)]
    out += [make(list(c), 401 + i, "test") for i, c in enumerate(COMBOS)]
    return out


def demo() -> dict:
    """The incident used in the demo video: a silent thermal straggler at 2 a.m.

    A fleet-wide fluent-bit rollout lands eight minutes earlier (the red
    herring an eager agent blames), and an unrelated GPU runs warm.
    """
    return {
        "id": "demo-thermal", "seed": 7, "kinds": ["thermal"], "split": "demo", "heldout": False,
        "faults": [{"kind": "thermal", "t0": 2647.0, "node": node_name(17), "gpu": 3, "sev": 1.0}],
        "herrings": [{"kind": "fluentbit", "t": 2160.0},
                     {"kind": "warm_gpu", "node": node_name(41), "gpu": 6},
                     {"kind": "old_xid13", "node": node_name(52), "gpu": 2}],
    }


def demo_sticky() -> dict:
    """Demo incident: a canary daemonset caps one node's GPUs at 450 W, and
    re-applies the cap if someone only resets the power limit. A quick check
    after the obvious fix passes; a 200-step definition of done does not."""
    return {
        "id": "demo-sticky", "seed": 11, "kinds": ["power_cap"], "split": "demo", "heldout": False,
        "faults": [{"kind": "power_cap", "t0": 2647.0, "node": node_name(23), "gpu": None, "sev": 1.0,
                    "extra": {"cap": 450, "reapply_s": 480}}],
        "herrings": [{"kind": "fluentbit", "t": 2160.0},
                     {"kind": "warm_gpu", "node": node_name(41), "gpu": 6},
                     {"kind": "old_xid13", "node": node_name(52), "gpu": 2}],
    }


DEMOS = {"demo-thermal": demo, "demo-sticky": demo_sticky}


def by_id(sid: str) -> dict:
    if sid in DEMOS:
        return DEMOS[sid]()
    for s in dev_set() + test_set():
        if s["id"] == sid:
            return s
    raise KeyError(sid)
