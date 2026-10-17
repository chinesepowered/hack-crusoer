# Capy FAQ

Quick answers to the questions we expect from judges. Every number comes from recorded runs in [results/](results/). Unless a question says otherwise, they're the 34 unseen test incidents that both agents finished.

## Capy in numbers

- **0 false "done" claims** across 52 dev and test incidents. A single agent with the same model and the same tools made one, and left a 512-GPU job running at 78% after closing the incident.
- **0 wasted disruptive actions** on the test set. The single agent drained a healthy node over a stale log line.
- **7 incomplete fixes caught** by Capy's own verification on the test set. All 7 ended in a verified fix.
- **50 vs 142 GPU-hours** lost on the two-fault incident, nearly 3× less than the single agent.
- **209 vs 313 GPU-hours** on each agent's worst incident: Capy's worst night is a third cheaper.
- **97% of unseen incidents fixed**, including 8 of the 9 incidents whose fault types no prompt mentions.
- **5.2¢ of inference per incident** on Crusoe, with 55% of prompt tokens served from Crusoe's prefix cache.
- **4 models, 1 endpoint:** Nemotron 3.5 Super, Nemotron 3 Nano and Nemotron 3.5 Lightning, plus Qwen 3.8 as the skeptic, all on Crusoe Managed Inference.

## The problem and the product

### What is Capy, in one sentence?
An on-call team of AI agents for multi-node training jobs on Crusoe. It finds the GPU, link or config change that slowed the whole job down, proves it, fixes it, and only says "done" after measuring that the job recovered.

### Is this a real problem?
Yes, and an expensive one. Meta's Llama 3 run had 419 unexpected interruptions in 54 days on 16,384 GPUs, roughly one every three hours, and about 78% were hardware. The costliest are gray: a throttling GPU, a flapping InfiniBand link, a config change that moved checkpoints to slow storage. Synchronous training runs at the speed of its slowest rank, so one slow GPU slows all 512, and nothing alarms because nothing crashed. A 512-GPU job running 25% slow wastes about 128 GPU-hours every hour.

### Doesn't Crusoe already handle this with AutoClusters and Instance Health?
They handle hard failures at the node level. Instance Health flags XID errors, AutoClusters replaces dead nodes, and Active Health Checks test idle ones. Capy covers what they can't see: job-level slowdowns where every node still reports healthy. It's also the missing piece for automation. AutoClusters ships with remediation switched off, and Capy's proof plus a measured verification is the evidence a team needs to switch it on.

### Why an AI agent and not alert rules?
An alert can tell you throughput dropped. It can't tell you why: the answer is spread across DCGM metrics, InfiniBand counters, kernel and NCCL logs, the change log and per-rank step timings, and it's different every time. That multi-step investigation is exactly what LLM agents are good at. What they're usually bad at is knowing when they're done, and that's the part Capy makes deterministic.

### Who would buy this?
Two kinds of buyer:
- Any team running multi-node training or fine-tuning on rented GPUs, from labs to startups.
- Crusoe itself, as a managed feature next to AutoClusters and Command Center.

An incident puts about 100 GPU-hours at stake, and Capy spends about 5 cents of inference handling it.

### Why capybaras?
Capybaras are famously calm, which is exactly what you want on call. They also make a multi-agent system easy to follow: each capybara is one role on one model, so in the console you can watch who is doing what.

## Done means done

### What does "done" mean in Capy, concretely?
Before anyone investigates, the lead writes a contract. Policy floors can be tightened but never loosened:
- throughput at ≥ 97% of the 7-day baseline over 200 measured steps
- a root cause that explains ≥ 90% of the slowdown
- an independent skeptic who agrees

On top of that, verification requires zero new errors on the job's nodes, and Capy gets one approval ping at most. The incident closes only when every check passes.

### Couldn't the model just claim it verified?
No. Models only propose. The proof gate (a counterfactual on per-rank timings) and the 200-step verification are code. They read telemetry directly, and it's their results that mark the contract, not the model's words.

### Why 200 steps, 97% and 90%?
- **200 steps** is about 15 minutes and always spans two checkpoints, one every 100 steps, so a periodic stall can't hide between measurements. That's exactly the gap the single agent's 50-step check fell into.
- **97%** leaves room for normal step-time noise.
- **90% explained** means the main cause is found, and a second hidden fault keeps the case open.

Teams can tighten all three.

### Does the loop actually catch anything?
Constantly. On the test set:
- **Verification:** Capy's own check rejected its first fix 7 times, with throughput still at 75–90%. Each time it re-planned and reached a verified fix.
- **Proof gate:** sent the case back 4 times out of 50.
- **Skeptic:** objected 4 times out of 44. In `power_cap-204` it pointed out that a daemonset would re-apply the power cap, so a reset alone would be undone.
- **Governor:** caught 13 loops (repeated identical tool calls) and cut them off.

### What happens when Capy can't figure it out?
It escalates to a human with everything it found, instead of guessing. On the test set, Capy's only unfixed incident went that way. The single agent never escalated; its unfixed incident on the test set was a confident, wrong "done".

## Results and the benchmark

### Your fix rate is the same as the single agent's. What does Capy add?
A "done" you can trust. Fixing is table stakes: our tools are good enough that a single Nemotron 3.5 Super agent with them fixes 97% of incidents too. But that agent once measured 50 steps between two slow checkpoint saves, saw 101.5%, and declared the job "self-recovered". The job stayed at 78%, burning about 114 GPU-hours every hour after the incident was closed. It also drained a healthy node over a stale XID 13 log line. Capy did neither. If 1 in 34 "done"s is false, a human has to double-check every one of them, and you're back on call.

### Is 1 false "done" vs 0 enough evidence?
The benchmark proves the failure is real: a strong agent with good tools walked right into it. Capy's guarantee doesn't depend on the count. Its contract requires 200 measured steps, which always span two checkpoints, plus a ≥ 90% counterfactual, so a lucky 50-step window can't close an incident. And across all 52 dev and test incidents, Capy never claimed a false "done".

### Isn't the benchmark rigged? You wrote the simulator and the agent.
We built it so it couldn't be:
- **Unseen test split:** new seeds, two-fault incidents, and 3 fault types (checkpoint stall, CPU contention, PCIe downtraining) that appear in no prompt and no dev scenario.
- **Red herrings in the incidents:**
  - an unrelated daemonset rollout
  - an old XID 13
  - a warm-but-fine GPU
  - slowly rising InfiniBand symbol errors
  - a degraded hot spare
  - an eval job on shared storage
- **Hidden ground truth:** "fixed" is judged by the simulator 75 minutes after the page, never by the agent.
- **Same setup for both:** the baseline gets the same model, the same tools and the same approval rule.
- **Every run published:** we tuned on dev only, and every run, win or loss, is in `results/`.

### Why is the baseline so strong?
Because we wanted Capy to beat the best version of a single agent, not a strawman. It gets the same model and every tool Capy has: the per-rank flight recorder, the counterfactual and the verifier. Anything Capy wins is won by the loop itself.

### How does it do on fault types it has never seen?
No prompt mentions checkpoint stalls, CPU contention or PCIe downtraining. Capy fixed 8 of the 9 held-out incidents with zero false "done" claims. The ninth it escalated rather than guess. The single agent also fixed 8, and its ninth was the false "done".

### What about two faults at once?
That's where the loop shines. If removing the claimed cause doesn't explain the slowdown, the case stays open. On the two-fault incident both agents finished, Capy lost 50 GPU-hours and the single agent 142. Capy's first fix verified at 83%, so the loop sent it back; it found the second fault and fixed both.

### Why are some test incidents unscored?
Six runs didn't finish. Three hit two harness bugs, both since fixed, and three stopped when our inference account hit a billing limit near the end of the sweep. We exclude them for both agents rather than count them for or against either, and list them in [results/test/summary.md](results/test/summary.md).

### What's the biggest limitation?
It runs on a high-fidelity simulator, not a live cluster yet. The tool interface is where Command Center's APIs plug in, and that's the next step. It also uses more inference than a single agent, about 5¢ against under 1¢ per incident, which is noise next to the ~100 GPU-hours at stake.

## Crusoe and Nemotron

### How do you use Crusoe?
Every agent call goes through Crusoe Managed Inference: one OpenAI-compatible endpoint, four models. We use tool calling, JSON mode, the per-call `enable_thinking` switch and the prefix cache, which served 55% of Capy's prompt tokens. On the product side, every Capy tool maps onto a Crusoe surface:
- Command Center metrics and logs
- Instance Health
- VM diagnostics and support tickets
- AutoClusters / CMK node replacement
- audit logs

The README has the full mapping.

### Why Nemotron?
1. **The right size for each job.** Super runs the long tool-calling investigation. Nano, a 30B MoE with 3B active parameters, runs four sweeps in parallel for a fraction of a cent. Lightning writes the case files and runbooks fast.
2. **Open weights.** A customer can run the whole team on a Nemotron deployment inside their own Crusoe project, so cluster telemetry never leaves it.
3. **It gets cheaper over time.** Verified incidents become fine-tuning data for Lightning, so routine triage moves to the small model.

### What did you learn about the Nemotron models?
- **Super** was a dependable tool-caller across investigations of 10–30 calls. With thinking switched off, it returned fast, well-formed JSON for the contract.
- **Nano** was the right size for parallel triage.
- **Lightning** occasionally returned a 503 "capacity unavailable", so the client falls back to Nano.
- **Qwen 3.8** reasons at length, so the skeptic gets a large token budget and a forced, thinking-off final verdict.

### If you're going for the Nemotron prize, why is the skeptic Qwen?
A skeptic from the same model family tends to share the proposer's blind spots, so falsification works better across families. The skeptic is the only non-Nemotron role; every other agent runs on Nemotron.

### What does it cost?
About 5 cents of inference per incident, roughly one H100-minute at typical rental prices, to protect around 100 GPU-hours.

### What would you need from Crusoe to ship this?
Most of it exists today:
- Command Center's Prometheus-compatible metrics and LogsQL logs
- Instance Health
- VM diagnostics
- AutoClusters node replacement
- audit logs

Two additions would make it native. One is an alert hook that triggers Capy (polling the metrics works to start). The other is a least-privilege role that allows Capy's short list of actions and nothing else.

## Safety

### What stops Capy from doing something destructive?
Several layers:
- Tools are read-only by default.
- Actions come from a short whitelist with schemas, and the plan is reduced to known actions and arguments before anything runs.
- Every fix has to pass the proof gate and the skeptic first.
- Disruptive actions (drain, restart, kill, pause) need a human's approval. The ping carries the proof, the skeptic's verdict and the GPU-hour cost.

On the test set Capy made zero wasted disruptive actions.

### What if nobody answers the ping?
In live mode Capy waits up to 15 minutes. No answer counts as a no: it doesn't act, and it escalates with the case file. In the benchmark, a simulated human approves after 3 minutes.

### What if a model is down?
The client retries 429s and 503s with backoff and falls back per role: Lightning and Qwen both fall back to Nano. If the skeptic can't answer, the deterministic gates still have to pass. If the lead can't be reached, the run stops and never claims done, so an outage can't produce a false "done".

### Can logs inject instructions into the agents?
Text alone can't trigger anything. A malicious log line can at most suggest a hypothesis. The plan is reduced to whitelisted actions, the proof gate checks it against telemetry, the skeptic reviews it, and anything disruptive needs a human.

## How it works

### Is this running on a real cluster?
Not yet. For a benchmark that's deliberate: you can't inject faults into someone's real 512-GPU job, and scoring "done" needs ground truth. The agents only see the simulator through the tool interface, which is exactly where Command Center's APIs plug in.

### How realistic is the simulator?
- **Timing:** it models synchronous training. Each step takes as long as the slowest rank's compute, plus exposed communication, data wait and overhead, with a checkpoint every 100 steps.
- **Faults look like they do on real hardware:**
  - NVML throttle-reason bits and clock drops
  - `nvidia-smi` power-limit messages
  - kernel XID 79, AER and PCIe downtraining lines
  - mlx5 link flaps and NCCL retries
  - `NCCL_IB_HCA` effects
  - slow checkpoint writes
- **Remediations cost what they would:**
  - a drain swaps in a hot spare and rolls back to the last checkpoint (7.5 minutes)
  - a restart takes 3.5 minutes
  - approvals take 3 minutes
  - every tool and model call costs simulated time while 512 GPUs run slow

### How does the proof gate work?
The flight recorder gives per-rank timings for recent steps. To test a claimed cause, Capy replaces that cause's contribution with its healthy value:
- that GPU's compute becomes the fleet median
- exposed comm returns to its 7-day p50
- data wait goes to 0
- checkpoint saves go back to their normal 4 s

It then recomputes the critical path and measures how much of the slowdown disappears. Below 90%, the case stays open.

### Couldn't the lead game the gate by blaming everything?
The gate checks that the claimed causes are enough to explain the slowdown. Three other checks keep the claim minimal:
- Each cause has to map to a concrete fix.
- The skeptic is asked whether a cheaper fix exists.
- An over-broad claim means a bigger disruptive fix, which a human sees on the ping, cost included.

### What's the shared brain?
After every verified fix, Archivist Capy writes a runbook (symptom signature, proof, fix) into the team's shared brain, and later incidents load those runbooks into the lead's prompt. Only fixes that were measured to work become lessons. It's seeded from the dev incidents, and measuring its lift on the test set is a one-flag experiment: `--brain brain/runbooks.jsonl`.

### How long does an incident take?
The median incident is verified fixed 29 minutes after the page, in simulated time. About 15 of those minutes are the 200-step verification itself. Without Capy, the alternative is a human noticing in the morning. A live run takes 2–8 minutes of wall-clock time, mostly model latency.

## Demo and logistics

### Can we see it live?
Yes. In the console, pick an incident and press "Run live on Crusoe". The agents run on Crusoe Managed Inference in real time, and you approve the ping yourself. Replays of recorded runs, which are the default view and what the video shows, work without an API key.

### What did you build during the hackathon?
Everything in this repo: the simulator, the agent team, the benchmark, the console, the video pipeline and the diagrams.

### Why didn't you use You.com search?
Incidents are answered from the job's own telemetry. Sending cluster details to a web search would add latency and leak data. A good future use is looking up an unfamiliar XID code or driver erratum, with the query scrubbed.

### What's next?
1. Point the read-only tools at Command Center on a live job.
2. Package Capy as a Command Center add-on, next to AutoClusters.
3. Fine-tune Nemotron 3.5 Lightning on verified incidents, so routine triage gets cheaper every week.
4. Run a bigger benchmark with more two-fault incidents, and measure the shared brain's lift.
5. Send the ping to Slack or PagerDuty.
