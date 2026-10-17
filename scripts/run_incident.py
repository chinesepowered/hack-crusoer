"""Run one incident with Capy or the baseline and save its trace.

    python scripts/run_incident.py demo-thermal --agent capy --out traces/demo-capy.json
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capy import scenarios  # noqa: E402
from capy.agents import Capy, run_baseline  # noqa: E402
from capy.brain import Brain  # noqa: E402
from capy.llm import LLM  # noqa: E402


def printer(ev):
    t = ev["type"]
    if t == "tick":
        return
    if t == "say":
        print(f"[{ev['clock']}] {ev['name']}: {ev['text'][:220]}")
    elif t == "tool":
        print(f"[{ev['clock']}]   {ev['who']} -> {ev['name']}({json.dumps(ev['args'])[:80]})")
    elif t in ("thinking",):
        return
    else:
        body = {k: v for k, v in ev.items() if k not in ("type", "t", "clock", "wall")}
        print(f"[{ev['clock']}] <{t}> {json.dumps(body)[:400]}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("--agent", choices=["capy", "baseline"], default="capy")
    ap.add_argument("--brain", default=None, help="runbook jsonl to load (read-only)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    sc = scenarios.by_id(a.scenario)
    llm = LLM()
    try:
        if a.agent == "capy":
            capy = Capy(llm, Brain(entries=Brain(a.brain).entries) if a.brain else None, learn=False)
            events = []
            res = await capy.run(sc, sink=(lambda e: (events.append(e), None if a.quiet else printer(e))))
        else:
            events = []
            res = await run_baseline(sc, llm, sink=(lambda e: (events.append(e), None if a.quiet else printer(e))))
    finally:
        await llm.close()
    print(json.dumps({k: v for k, v in res.items() if k not in ("usage", "report")}, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"scenario": sc, "result": res, "events": events}))
        print("saved", a.out)


asyncio.run(main())
