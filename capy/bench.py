"""Benchmark: Capy vs a single-agent baseline on simulated incidents.

    python -m capy.bench --split dev --agents capy,baseline --out results/dev
    python -m capy.bench --split test --agents capy,baseline --out results/test --brain brain/dev.jsonl

The test split adds new seeds, three fault types that appear in no prompt
and no dev scenario (held-out), and two-fault incidents.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from . import scenarios
from .agents import Capy, run_baseline
from .brain import Brain
from .llm import LLM


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    n = len(rows)
    claimed = [r for r in rows if r["claimed_done"]]
    def mean(key, rs=rows):
        return round(statistics.mean(r[key] for r in rs), 3) if rs else None
    return {
        "n": n,
        "fixed_pct": round(100 * sum(r["fixed"] for r in rows) / n, 1),
        "diagnosis_correct_pct": round(100 * sum(r["diagnosis_correct"] for r in rows) / n, 1),
        "claimed_done_pct": round(100 * len(claimed) / n, 1),
        "false_done_pct_of_claims": round(100 * sum(r["false_done"] for r in claimed) / len(claimed), 1) if claimed else 0.0,
        "false_done_count": sum(r["false_done"] for r in rows),
        "escalated_pct": round(100 * sum(r["outcome"] in ("escalated", "gave_up") for r in rows) / n, 1),
        "gpu_hours_lost_mean": mean("gpu_hours_lost"),
        "gpu_hours_lost_total": round(sum(r["gpu_hours_lost"] for r in rows), 1),
        "wasted_disruptive_actions": sum(r["wasted_disruptive_actions"] for r in rows),
        "disruptive_actions": sum(r["disruptive_actions"] for r in rows),
        "pings_per_incident": mean("pings"),
        "minutes_to_done_median": round(statistics.median(r["minutes_to_done"] for r in claimed), 1) if claimed else None,
        "tool_calls_mean": mean("tool_calls"),
        "cost_usd_mean": round(statistics.mean(r["cost_usd"] for r in rows), 4),
        "cost_usd_total": round(sum(r["cost_usd"] for r in rows), 3),
        "prompt_tokens_total": sum(r["prompt_tokens"] for r in rows),
        "cached_tokens_total": sum(r["cached_tokens"] for r in rows),
    }


def group(rows, keyf):
    out = {}
    for r in rows:
        out.setdefault(keyf(r), []).append(r)
    return out


def report(rows: list[dict]) -> tuple[dict, str]:
    agents = sorted(set(r["agent"] for r in rows))
    summary = {"overall": {a: summarize([r for r in rows if r["agent"] == a]) for a in agents}}
    cats = {"seen single faults": lambda r: not r["heldout"] and len(r["kinds"]) == 1,
            "held-out fault types": lambda r: r["heldout"] and len(r["kinds"]) == 1,
            "two-fault incidents": lambda r: len(r["kinds"]) > 1}
    summary["by_category"] = {c: {a: summarize([r for r in rows if r["agent"] == a and f(r)]) for a in agents}
                              for c, f in cats.items()}
    summary["by_kind"] = {k: {a: summarize([r for r in v if r["agent"] == a]) for a in agents}
                          for k, v in sorted(group(rows, lambda r: "+".join(r["kinds"])).items())}
    lines = ["| metric | " + " | ".join(agents) + " |", "|---|" + "---|" * len(agents)]
    labels = [("n", "incidents"), ("fixed_pct", "fixed (%)"), ("diagnosis_correct_pct", "correct root cause (%)"),
              ("claimed_done_pct", "claimed done (%)"), ("false_done_count", "false 'done' claims"),
              ("false_done_pct_of_claims", "false 'done' (% of claims)"), ("escalated_pct", "escalated / gave up (%)"),
              ("gpu_hours_lost_mean", "GPU-hours lost / incident"), ("wasted_disruptive_actions", "wasted disruptive actions"),
              ("pings_per_incident", "human pings / incident"), ("minutes_to_done_median", "minutes to done (median)"),
              ("tool_calls_mean", "tool calls / incident"), ("cost_usd_mean", "inference $ / incident")]
    for key, label in labels:
        lines.append(f"| {label} | " + " | ".join(str(summary["overall"][a].get(key)) for a in agents) + " |")
    md = ["## Overall", ""] + lines + ["", "## By category", ""]
    for c, per in summary["by_category"].items():
        md.append(f"**{c}**: " + "; ".join(
            f"{a}: fixed {per[a].get('fixed_pct')}%, correct {per[a].get('diagnosis_correct_pct')}%, false-done {per[a].get('false_done_count')}, "
            f"GPU-h lost {per[a].get('gpu_hours_lost_mean')}" for a in agents if per[a]))
    md += ["", "## By fault", "", "| fault | " + " | ".join(f"{a} fixed / correct / GPU-h" for a in agents) + " |",
           "|---|" + "---|" * len(agents)]
    for k, per in summary["by_kind"].items():
        md.append(f"| {k} | " + " | ".join(
            f"{per[a].get('fixed_pct')}% / {per[a].get('diagnosis_correct_pct')}% / {per[a].get('gpu_hours_lost_mean')}" if per[a] else "-"
            for a in agents) + " |")
    return summary, "\n".join(md)


def trace_error(path: Path) -> str:
    try:
        ev = json.loads(path.read_text())["events"]
        return next(e["text"] for e in ev if e.get("type") == "error")
    except (OSError, ValueError, KeyError, StopIteration):
        return "inference error"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--agents", default="capy,baseline")
    ap.add_argument("--only", default="", help="comma-separated scenario ids")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", default="results/dev")
    ap.add_argument("--brain", default=None, help="runbook jsonl for Capy (read-only)")
    ap.add_argument("--learn-brain", default=None, help="runbook jsonl Capy appends to after verified fixes")
    ap.add_argument("--traces", action="store_true")
    ap.add_argument("--summarize", action="store_true", help="only rebuild summary from runs.jsonl (latest run per scenario+agent)")
    a = ap.parse_args()
    scs = (scenarios.dev_set() if a.split in ("dev", "all") else []) + (scenarios.test_set() if a.split in ("test", "all") else [])
    if a.summarize:
        latest = {}
        for line in (Path(a.out) / "runs.jsonl").read_text().splitlines():
            r = json.loads(line)
            latest[(r["scenario"], r["agent"])] = r
        agents = a.agents.split(",")
        # Paired comparison: an incident counts only if every agent finished it.
        # Runs that never finished (harness crash, provider error) are listed, not scored.
        excluded = {}
        for sc in scs:
            for ag in agents:
                r = latest.get((sc["id"], ag))
                if r is None:
                    excluded.setdefault(sc["id"], []).append(f"{ag}: run crashed (harness bug), no result")
                elif r["outcome"] == "error":
                    why = r.get("error") or trace_error(Path(a.out) / "traces" / f"{sc['id']}-{ag}.json")
                    excluded.setdefault(sc["id"], []).append(f"{ag}: {why}"[:160])
        rows = [r for (sid, ag), r in latest.items() if ag in agents and sid not in excluded
                and sid in {sc["id"] for sc in scs}]
        summary, md = report(rows)
        summary["excluded"] = excluded
        if excluded:
            md += "\n\n## Not scored (unfinished runs, excluded for every agent)\n\n" + "\n".join(
                f"- {sid}: " + "; ".join(why) for sid, why in excluded.items())
        (Path(a.out) / "summary.json").write_text(json.dumps(summary, indent=1))
        (Path(a.out) / "summary.md").write_text(md + "\n")
        print(md)
        return
    if a.only:
        keep = set(a.only.split(","))
        scs = [s for s in scs if s["id"] in keep]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    llm = LLM()
    brain = Brain(a.learn_brain) if a.learn_brain else Brain(entries=Brain(a.brain).entries if a.brain else [])
    capy = Capy(llm, brain, learn=bool(a.learn_brain))
    sem = asyncio.Semaphore(a.concurrency)
    rows = []
    t0 = time.time()

    async def one(sc, agent):
        async with sem:
            events = []
            try:
                if agent == "capy":
                    r = await capy.run(sc, sink=events.append)
                else:
                    r = await run_baseline(sc, llm, sink=events.append)
            except Exception as e:  # keep the sweep going; record the failure
                r = {"scenario": sc["id"], "agent": agent, "error": repr(e)}
            if "error" in r and "fixed" not in r:
                print(f"!! {sc['id']} {agent}: {r['error']}", flush=True)
                return
            rows.append(r)
            with (out / "runs.jsonl").open("a") as f:
                f.write(json.dumps({k: v for k, v in r.items() if k != "usage"}) + "\n")
            if a.traces:
                (out / "traces").mkdir(exist_ok=True)
                (out / "traces" / f"{sc['id']}-{agent}.json").write_text(json.dumps({"scenario": sc, "result": r, "events": events}))
            print(f"{time.time() - t0:6.0f}s {sc['id']:32s} {agent:8s} {r['outcome']:9s} fixed={r['fixed']!s:5s} "
                  f"correct={r['diagnosis_correct']!s:5s} false_done={r['false_done']!s:5s} gpuh={r['gpu_hours_lost']:6.1f} "
                  f"${r['cost_usd']:.3f}", flush=True)

    agents = a.agents.split(",")
    await asyncio.gather(*[one(sc, ag) for sc in scs for ag in agents])
    await llm.close()
    summary, md = report(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "summary.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    asyncio.run(main())
