# Capy: an AI on-call team for GPU training runs on Crusoe

Capy is a team of AI capybaras that babysits multi-node GPU training jobs on Crusoe while you sleep. When a job slows down, Capy finds the cause, proves it, fixes it, and only says "done" after it has measured the recovery. Every agent runs on Crusoe Managed Inference: three Nemotron models, plus Qwen 3.8 as an independent skeptic. It wakes you only to approve a disruptive fix, or when it can't prove a cause.

<p align="center"><a href="https://www.youtube.com/watch?v=RyOno2s7pEo"><img src="docs/demo-thumbnail.jpg" width="900" alt="Capy console replaying a real incident. Click to watch the 2-minute demo on YouTube"></a></p>
<p align="center"><b><a href="https://www.youtube.com/watch?v=RyOno2s7pEo">▶ Watch the 2-minute demo</a></b> · <a href="FAQ.md">Judges' FAQ</a> · Crusoe × NVIDIA hackathon, <i>Done Means Done</i> track</p>

## Problem

A synchronous training job runs at the speed of its slowest GPU. The expensive failures are gray, not hard: one throttling GPU, a flapping InfiniBand link or a bad config change slows all 512 GPUs by 10–30% while every health check stays green. That's about 128 GPU-hours lost every hour until someone notices in the morning. And an AI agent pointed at the problem will happily say "done" when it isn't.

## Solution

Capy treats "done" as something you measure, not something a model says. Before anyone investigates, Detective Capy (Nemotron 3.5 Super) writes the contract: throughput back to ≥ 97% of baseline for 200 measured steps, a root cause that explains ≥ 90% of the slowdown, and an independent skeptic who agrees. Four Field Capys (Nemotron 3 Nano) sweep GPUs, network, storage and recent changes in parallel, and the detective works the case with read-only tools.

Every diagnosis has to survive two checks. First, a deterministic proof gate removes the claimed cause from per-rank step timings and recomputes the critical path. Then Inspector Doubt, running Qwen 3.8 so its blind spots differ, tries to falsify the cause and the fix. Safe fixes run right away. Disruptive ones send you one ping with the proof and the GPU-hour cost attached. The case closes only after 200 real steps verify the recovery, and any failed check sends Capy back to investigate.

On unseen test incidents, Capy matched a strong single agent's 97% fix rate with zero false "done" claims and zero wasted drains or restarts, and its own checks caught 7 incomplete fixes before anyone could call them done. Archivist Capy (Nemotron 3.5 Lightning) turns every verified fix into a runbook for the team's shared brain.

> AutoClusters replaces dead GPUs. Capy finds the slow ones, and proves it.

## Architecture

<p align="center"><img src="docs/architecture.svg" width="100%" alt="Architecture: the training job on Crusoe feeds telemetry to Capy's eight-step loop (contract, triage, investigate, proof gate, skeptic, fix, verify, close), which calls Crusoe Managed Inference for every agent, pings the human at most once, and writes runbooks to a shared brain"></p>

- **Left, your training job on Crusoe.** Capy reads telemetry through read-only tools and acts through a short list of remediations. Here a simulator of a 512× H100 cluster, with injected faults and ground truth, stands in for it.
- **Middle, the loop.** LLM steps propose; deterministic gates decide, so no agent can talk its way past them.
- **Right, Crusoe Managed Inference.** It serves every agent. Verified fixes become runbooks, and the web console streams the whole case live.

| Step | Agent | What has to be true to move on |
|---|---|---|
| 1. Contract | Detective Capy | A definition of done exists before anyone investigates: 200 steps at ≥ 97% of baseline, a cause that explains ≥ 90% of the slowdown, a skeptic who agrees, no new errors, at most one ping. Policy floors can be tightened, never loosened. |
| 2. Triage | 4 Field Capys | GPUs, network, data pipeline and recent changes are swept in parallel. Findings are leads, not conclusions. |
| 3. Investigate | Detective Capy | Hypotheses are tested with read-only tools and tracked on a case board. |
| 4. Proof gate | code | Remove the claimed cause from the per-rank flight recorder and recompute the critical path. Below 90% explained means keep looking, which is how a second, hidden fault gets caught. |
| 5. Skeptic | Inspector Doubt | A different model family tries to falsify the diagnosis and the fix (is there a cheaper safe fix? are the arguments right?). |
| 6. Fix | Capy | Safe fixes run immediately. Disruptive ones (drain, restart, kill, pause) send one ping with the proof, the skeptic's verdict and the GPU-hour cost attached. |
| 7. Verify | code | 200 real steps are measured after the fix takes effect. A failure means re-plan, never a victory lap. |
| 8. Close | Archivist Capy | The case file and a runbook go into the team's shared brain. Only verified fixes become lessons. |

## How we leverage Crusoe

### Crusoe Managed Inference runs every agent

| Agent | Model on Crusoe | Why this model |
|---|---|---|
| Detective Capy: lead, contract | `nvidia/NVIDIA-Nemotron-3.5-Super-VL` | The strongest reasoning and tool calling, for an investigation that takes 10–30 tool calls. |
| 4 Field Capys: triage | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` | A small MoE (3B active), so four sweeps run in parallel for a fraction of a cent. |
| Archivist Capy: case file, runbook | `nvidia/Nemotron-3.5-Lightning-30B-A3B` | A fast writer for summaries. Falls back to Nano when capacity is busy. |
| Inspector Doubt: skeptic | `Qwen/Qwen3.8-27B` | The only non-Nemotron model, on purpose: a different family has different blind spots. |

- **Tool calling and JSON mode** on every agent, and the `enable_thinking` switch set per call: reasoning on for investigation and triage, off where we need fast structured output (the contract, the skeptic's verdict, the case file).
- **Prefix caching:** stable system prompts and the shared brain sit at the front of every prompt. On the test sweep, 55% of Capy's prompt tokens were served from Crusoe's cache.
- **Throughput:** four triage calls in parallel per incident and a 12-way concurrent benchmark, inside the account's 500k tokens/min limit. The client retries 429s and 503s with backoff, falls back per role, and accounts tokens, cache hits and cost per run.
- **Cost:** 5.2¢ of inference per incident on average, against roughly 100 GPU-hours at stake.

### Capy extends the platform it runs on

Crusoe already heals hard failures: Instance Health flags XID errors, AutoClusters replaces dead nodes, Active Health Checks test idle ones. Capy is the job-level layer for failures that are slow but alive. It also brings the evidence a team needs before switching automation on; AutoClusters ships with remediation off by default. Every Capy tool maps onto a Crusoe surface:

| Capy tool | Production source on Crusoe |
|---|---|
| `gpu_outliers`, `gpu_detail`, `node_detail` | Command Center metrics (DCGM fields) through the Prometheus-compatible API |
| `search_logs` | Command Center logs (JournalD, including XID lines) through LogsQL |
| `fleet_health` | Command Center Instance Health |
| `collect_diagnostics`, `file_support_ticket` | `crusoe diagnostics vm create` bug reports and Crusoe support |
| `drain_node` | AutoClusters / Crusoe Managed Kubernetes node replacement with hot spares |
| `change_log` | Crusoe audit logs, plus your CI/CD and ArgoCD history |
| `rank_timings`, `explain_slowdown` | the PyTorch / NCCL flight recorder exported by the trainer |

- **Telemetry can stay in the customer's project:** Nemotron is open, so the whole team can run on a self-serve Nemotron deployment inside the customer's own Crusoe project.
- **Next:** point the read-only tools at Command Center on a real job, and turn verified incidents into a fine-tuning set for Nemotron 3.5 Lightning with Crusoe's serverless fine-tuning, so routine triage moves to the small model.

## Results

Both agents run live on Crusoe Managed Inference against simulated incidents, scored by the simulator's ground truth 75 minutes after the page. The baseline is a single Nemotron 3.5 Super agent with **the same tools** and the same approval rule, so the comparison isolates Capy's loop.

**Test split** (new seeds, 3 fault types that no prompt or dev scenario mentions, two-fault incidents): 34 incidents scored for both agents. Full tables are in [results/test/summary.md](results/test/summary.md).

| | Capy | single agent |
|---|---|---|
| job actually fixed | 97.1% | 97.1% |
| false "done" claims | **0** | 1 |
| wasted disruptive actions | **0** | 1 |
| correct root cause | 94.1% | 97.1% |
| escalated to a human | 1 | 0 |
| GPU-hours lost / incident | 101.6 | 100.7 |
| human pings / incident | 0.76 | 0.74 |
| inference cost / incident | 5.2¢ | 0.7¢ |

- **Same fix rate, but only one of them is right when it says "done".** The single agent once measured 50 steps that fell between two slow checkpoint saves, saw 101.5%, and declared the job "self-recovered". It stayed at 78%, losing about 114 GPU-hours every hour after the incident was closed. That run opens the video. It also drained a healthy node over a stale XID 13 log line. Capy did neither, and across all 52 dev and test incidents it never claimed a false "done".
- **The loop earns its keep.** On the test set, Capy's own verification rejected its first fix 7 times, and every one ended in a verified fix. The proof gate sent 4 cases back, the skeptic objected to 4 proposals (one fix would have been undone by the daemonset that set the power cap), and the governor stopped 13 loops.
- **Two faults at once:** Capy lost 50 GPU-hours, the single agent 142, nearly 3× more.
- **When Capy can't prove a cause, it escalates.** Its one unfixed test incident went to a human with the evidence, not into a false "done".
- **Scoring notes:** six test incidents with an unfinished run (harness bugs since fixed, and our inference account hitting a billing limit) are excluded for both agents and listed in the [full results](results/test/summary.md). The dev split was used while building: [results/dev/summary.md](results/dev/summary.md).

## Why it's a product

- **Buyer:** any team running multi-node training or fine-tuning on rented GPUs, and Crusoe itself, as a managed feature next to AutoClusters and Command Center.
- **ROI:** a 512-GPU job running 25% slow loses about 128 GPU-hours every hour. Capy costs cents of inference per incident, and its median incident is verified fixed 29 minutes after the page, not the next morning.
- **Trust is the product:** every fix comes with a proof and a skeptic's verdict, and is measured before anyone calls it done, so a team can let Capy act while they sleep.

## Run it

```bash
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
uvicorn capy.server:app --port 8765    # open http://localhost:8765 and press ▶ Replay (no API key needed)
```

Live runs and the benchmark need a Crusoe Managed Inference key:

```bash
cp .env.example .env                   # add CRUSOE_API_KEY
python scripts/smoke_test.py           # checks tool calling, JSON mode, the thinking switch and rate limits
python scripts/run_incident.py demo-sticky --agent capy                          # one incident in the terminal
python -m capy.bench --split test --agents capy,baseline --out results/test --traces
python -m capy.bench --split test --out results/test --summarize                # rebuild the paired scoreboard
```

In the console, "Run live on Crusoe" starts a real run and you approve the ping yourself. The scoreboard is at `/results`.

## Repo map

```
capy/sim.py        512× H100 cluster + training job simulator: DCGM metrics, IB counters, JournalD/NCCL/trainer logs,
                   change log, Instance Health, per-rank flight recorder; fault injection, remediation, ground truth
capy/scenarios.py  dev set, test set (new seeds, 3 held-out fault types, two-fault incidents), demo incidents
capy/tools.py      the agents' read-only tools and actions, each with a simulated time cost
capy/agents.py     the Capy team, the governor, the proof gate, and the single-agent baseline
capy/llm.py        Crusoe Managed Inference client: retries, fallbacks, token, cache and cost accounting
capy/bench.py      benchmark runner and paired scoreboard
capy/server.py     FastAPI: live runs over SSE, approvals, trace replay, results
web/               the console (vanilla JS, hand-drawn SVG capybaras) and the scoreboard page
traces/            recorded incidents the console replays
video/             demo video pipeline (ElevenLabs narration, Playwright recording, ffmpeg)
```

## Scope and next steps

- **A high-fidelity simulator today.** Failure signatures follow real behavior (NVML throttle reasons, XID 79 and PCIe downtraining lines, mlx5 link flaps, `NCCL_IB_HCA` effects, synchronous critical-path timing), every remediation costs what it would, and agents never see the hidden fault list. Next: point the read-only tools at Command Center on a live job.
- **A deliberately tough baseline.** The single agent gets Capy's tools, so the comparison measures the loop, not the tooling.
- **More to measure:** a bigger benchmark with more two-fault incidents, the shared brain's lift, and a Nemotron 3.5 Lightning fine-tune on verified incidents.
