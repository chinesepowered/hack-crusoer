"""Web server: live incident runs over SSE, approvals, trace replay, results.

    uvicorn capy.server:app --port 8000
"""
from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import scenarios
from .agents import APPROVAL_S, Capy, run_baseline
from .brain import Brain
from .llm import LLM

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
TRACES = ROOT / "traces"
RESULTS = ROOT / "results"
BRAIN = ROOT / "brain" / "runbooks.jsonl"

app = FastAPI(title="Capy")
app.mount("/static", StaticFiles(directory=WEB), name="static")
RUNS: dict[str, dict] = {}
_llm: LLM | None = None


def llm() -> LLM:
    global _llm
    if _llm is None:
        _llm = LLM()
    return _llm


class RunReq(BaseModel):
    scenario: str = "demo-thermal"
    agent: str = "capy"


class Approval(BaseModel):
    approved: bool = True


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/results")
def results_page():
    return FileResponse(WEB / "results.html")


@app.get("/api/scenarios")
def list_scenarios():
    out = [{"id": k, "kinds": v()["kinds"], "split": "demo"} for k, v in scenarios.DEMOS.items()]
    out += [{"id": s["id"], "kinds": s["kinds"], "split": s["split"]} for s in scenarios.dev_set()]
    return out


@app.get("/api/traces")
def list_traces():
    return sorted(p.stem for p in TRACES.glob("*.json"))


@app.get("/api/trace/{name}")
def get_trace(name: str):
    p = TRACES / f"{Path(name).name}.json"
    if not p.exists():
        raise HTTPException(404, f"no trace {name}")
    return JSONResponse(json.loads(p.read_text()))


@app.get("/api/results/{split}")
def get_results(split: str):
    p = RESULTS / Path(split).name / "summary.json"
    if not p.exists():
        raise HTTPException(404, f"no results for {split}")
    return JSONResponse(json.loads(p.read_text()))


@app.post("/api/run")
async def start_run(req: RunReq):
    try:
        sc = scenarios.by_id(req.scenario)
    except KeyError:
        raise HTTPException(404, f"unknown scenario {req.scenario}")
    rid = uuid.uuid4().hex[:8]
    state = {"events": [], "subs": [], "approval": None, "done": False, "result": None}
    RUNS[rid] = state

    def sink(ev):
        state["events"].append(ev)
        for q in state["subs"]:
            q.put_nowait(ev)

    async def approve(request, run):
        fut = asyncio.get_running_loop().create_future()
        state["approval"] = fut
        try:
            ok = await asyncio.wait_for(fut, timeout=900)
        except asyncio.TimeoutError:
            ok = False
        state["approval"] = None
        run.cluster.advance(APPROVAL_S)
        return ok

    async def go():
        try:
            if req.agent == "baseline":
                state["result"] = await run_baseline(sc, llm(), sink=sink, approve=approve)
            else:
                capy = Capy(llm(), Brain(BRAIN), learn=True)
                state["result"] = await capy.run(sc, sink=sink, approve=approve)
        except Exception as e:  # surface failures to the UI
            sink({"type": "error", "text": repr(e), "t": 0, "clock": "", "wall": 0})
        finally:
            state["done"] = True
            for q in state["subs"]:
                q.put_nowait(None)

    asyncio.create_task(go())
    return {"run_id": rid}


@app.get("/api/stream/{rid}")
async def stream(rid: str):
    state = RUNS.get(rid)
    if state is None:
        raise HTTPException(404, "unknown run")
    q: asyncio.Queue = asyncio.Queue()
    for ev in state["events"]:
        q.put_nowait(ev)
    if state["done"]:
        q.put_nowait(None)
    else:
        state["subs"].append(q)

    async def gen():
        while True:
            ev = await q.get()
            if ev is None:
                yield "event: end\ndata: {}\n\n"
                break
            yield f"data: {json.dumps(ev)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/approve/{rid}")
async def approve_run(rid: str, body: Approval):
    state = RUNS.get(rid)
    if not state or not state["approval"] or state["approval"].done():
        raise HTTPException(409, "nothing to approve")
    state["approval"].set_result(body.approved)
    return {"ok": True}
