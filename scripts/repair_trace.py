"""Rebuild per-event definition-of-done statuses in traces recorded before the
emitter snapshotted payloads (contract dicts were mutated after emission).

Replays Capy's own marking rules over the event stream, so each `contract`
and `contract_update` event shows the statuses that held at that moment.

    python scripts/repair_trace.py traces/demo-checkpoint-capy.json
"""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capy.agents import POLICY, Capy  # noqa: E402

KEYS = Capy.KEYS


def mark(items, kind, status, note=""):
    for d in items:
        text = (d.get("check", "") + " " + d.get("verified_by", "")).lower()
        if d.get("verified_by") == kind or any(k in text for k in KEYS.get(kind, ())):
            if d.get("status") == "pass" and status != "pass" and kind == "verify_recovery":
                continue
            d["status"] = status
            d["note"] = note[:200] if note else d.get("note", "")


def repair(path):
    tr = json.loads(Path(path).read_text())
    items, pings, hardware = None, 0, False
    for ev in tr["events"]:
        t = ev["type"]
        if t == "contract":
            items = copy.deepcopy(ev["contract"]["done_when"])
            for d in items:
                d["status"], d["note"] = "open", ""
            ev["contract"]["done_when"] = copy.deepcopy(items)
            continue
        if items is None:
            continue
        if t == "proof" and not ev.get("healthy"):
            mark(items, "explain_slowdown", "pass" if ev["ok"] else "fail", ev.get("summary", ""))
        elif t == "proposal":
            hardware = any(r.get("type") in ("gpu_thermal", "gpu_hardware_failure", "ib_link", "pcie_link")
                           for r in ev["proposal"].get("root_causes") or [])
        elif t == "skeptic":
            mark(items, "skeptic", "pass" if ev["upheld"] else "fail", "; ".join(ev.get("objections") or [])[:200])
        elif t == "ping":
            pings += 1
        elif t == "verify":
            mark(items, "verify_recovery", "pass" if ev["pass"] else "fail", ev.get("summary", ""))
        elif t == "ticket":
            mark(items, "ticket", "pass")
        elif t == "closed":
            if not hardware:
                mark(items, "ticket", "pass", "n/a: no hardware fault")
            mark(items, "ping", "pass" if pings <= POLICY["max_pings"] else "fail", f"{pings} ping(s)")
        elif t == "contract_update":
            ev["done_when"] = copy.deepcopy(items)
            continue
        else:
            continue
        # insert the updated state right where it changed
        ev.setdefault("_contract_after", copy.deepcopy(items))
    # fold the snapshots into the following contract_update events
    last = None
    updates = []
    for ev in tr["events"]:
        if "_contract_after" in ev:
            last = ev.pop("_contract_after")
        elif ev["type"] == "contract_update" and last is not None:
            ev["done_when"] = last
            updates.append(ev)
    # close-time marks (ping budget, ticket n/a) land on the final update
    if updates and items is not None:
        updates[-1]["done_when"] = copy.deepcopy(items)
    Path(path).write_text(json.dumps(tr))
    print("repaired", path)


for p in sys.argv[1:]:
    repair(p)
