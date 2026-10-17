# Capy FAQ

Answers to the questions we expect from judges. Numbers come from the benchmark in [results/](results/), mostly the 34 test incidents both agents finished, unless a different set is named.

## The problem and the product

### What is Capy, in one sentence?
An on-call team of AI agents that watches multi-node training jobs on Crusoe, finds the GPU, link or config change that slowed the whole job down, proves it, fixes it, and only says "done" after measuring that the job recovered.

### Is this a real problem?
Yes. Meta's Llama 3 run had 419 unexpected interruptions in 54 days on 16,384 GPUs, about one every three hours, and roughly 78% were hardware. The expensive ones are often gray: a throttling GPU, a flapping InfiniBand link, a config change that moved checkpoints to slow storage. Synchronous training runs at the speed of its slowest rank, so one slow GPU slows all 512, and since nothing crashes, nothing alarms until a human looks at a throughput chart. A 512-GPU job running 25% slow wastes about 128 GPU-hours every hour.

### Doesn't Crusoe already handle this with AutoClusters and Instance Health?
Those handle hard failures at the node level. Instance Health flags XID errors, AutoClusters replaces dead nodes, and Active Health Checks test idle ones. Capy works at the job level, on failures where every node still reports healthy. The two fit together: Capy's proof (a counterfactual plus a measured verification) is the evidence a team needs before it turns on automated remediation, which AutoClusters ships with switched off.

### Why an AI agent and not alert rules?
An alert can tell you throughput dropped. It can't tell you why. The answer is spread across DCGM metrics, InfiniBand counters, kernel and NCCL logs, the change log and per-rank step timings, and it's different every time. That's a multi-step investigation, which LLM agents are good at. What they're bad at is knowing when they're actually done, and Capy makes that part deterministic.

### Who would buy this?
Teams running multi-node training or fine-tuning on rented GPUs, from labs to startups, and Crusoe itself, as a managed feature next to AutoClusters and Command Center. An incident puts about 100 GPU-hours at stake, and Capy spends about 5 cents of inference handling it.

### Why capybaras?
Capybaras are famously calm, which is what you want on call. They also make a multi-agent system easy to follow: each capybara is one role on one model, so in the console you can see who is doing what.

## Done means done

### What does "done" mean in Capy, concretely?
Before anyone investigates, the lead writes a contract. Policy sets floors it can tighten but never loosen: throughput at ≥ 97% of the 7-day baseline over 200 measured steps, a root cause that explains ≥ 90% of the slowdown, and a skeptic who agrees. Verification also requires zero new errors on the job's nodes, and Capy may ping a human at most once. The incident closes only when every check passes.

### Couldn't the model just claim it verified?
No. Models only propose. The proof gate (a counterfactual on per-rank timings) and the 200-step verification are code. They read telemetry directly, and their results, not the model's words, mark the contract.

### Why 200 steps, 97% and 90%?
- **200 steps** is about 15 minutes and always spans two checkpoints (one every 100 steps), so a periodic stall can't hide between measurements. That's exactly how 50 steps fooled the single agent.
- **97%** leaves room for normal step-time noise.
- **90% explained** means the main cause is found, while a second hidden fault still keeps the case open.

Teams can tighten all three.

### Does the loop actually catch anything?
Yes. On the test set:
- Capy's own verification rejected its first fix in 7 incidents, with throughput still at 75–90%. Each time it re-planned and reached a verified fix.
- The proof gate sent the case back 4 times out of 50.
- The skeptic objected 4 times out of 44. In `power_cap-204`, for example, it pointed out that a daemonset would re-apply the power cap, so a reset alone would be undone.
- The governor caught 13 loops (repeated identical tool calls) and cut them off.

### What happens when Capy can't figure it out?
It escalates to a human with everything it found instead of guessing. That happened once on the test set, on a held-out checkpoint stall. The single agent never escalated, and once declared done on a job that was still 22% slow.

## Results and the benchmark

### Your fix rate is the same as the single agent's. What does Capy add?
Trust. With the same model and the same tools, a single Nemotron 3.5 Super agent fixed 97% of incidents, the same as Capy. But it once declared a job "self-recovered" after measuring 50 steps that happened to fall between two slow checkpoint saves. The job stayed at 78%, losing about 114 GPU-hours every hour after the agent closed the incident. It also drained a healthy node over a stale XID 13 log line. Capy made zero false "done" claims and zero wasted disruptive actions. If 1 in 34 "done"s is false, someone has to double-check every one, and you're back on call.

### 1 vs 0 false "done" claims isn't statistically significant.
Agreed, and the README says so: 34 scored test incidents is a small sample. The stronger argument is structural. That failure can't pass Capy's contract, because a 50-step check fails the 200-step rule, and a 200-step window always contains a checkpoint save.

### Isn't the benchmark rigged? You wrote the simulator and the agent.
We worked to keep it fair:
- **Unseen test split:** new seeds, two-fault incidents, and 3 fault types that appear in no prompt and no dev scenario: checkpoint stall, CPU contention, PCIe downtraining.
- **Red herrings in the incidents:** an unrelated daemonset rollout, an old XID 13, a warm-but-fine GPU, slowly rising InfiniBand symbol errors, a degraded hot spare, an eval job on shared storage.
- **Independent scoring:** "fixed" is judged by the simulator's hidden ground truth at a fixed time after the page, never by the agent.
- **Same setup for the baseline:** the same model, the same tools and the same approval rule.
- **Honest reporting:** we tuned on dev only, report where Capy lost (root-cause accuracy 94% vs 97%, 8× the inference cost, slightly more GPU-hours on seen faults), and list the runs that didn't finish.

### Why is the baseline so strong?
On purpose. It gets Capy's tools, including the per-rank flight recorder, the counterfactual tool and the verifier. Without them it would look much worse, but then we'd be measuring the tools, not the loop.

### What happened to the 6 unscored test incidents?
Three Capy runs crashed on two harness bugs, both since fixed: twice the lead called `kill_process` without a pid, and once a model sent a list where the harness expected a single number. Three more were cut off near the end of the sweep when our Crusoe account started returning HTTP 402 (billing). An incident is scored only if both agents finished it, so these are excluded for both and listed in [results/test/summary.md](results/test/summary.md). The command to re-run them is in the README.

### How does it do on fault types it has never seen?
On the 9 scored held-out incidents, both agents fixed 8. Capy named the right root cause in 7, the single agent in 8. The single agent's false "done" and Capy's one escalation both came from this group.

### What about two faults at once?
That's what the 90% rule is for: if removing the claimed cause doesn't explain the slowdown, the case stays open. Only one two-fault incident finished for both agents. Capy lost 50 GPU-hours there and the single agent 142. Capy's first fix verified at only 83%, so it went back, found the second fault and fixed it. That's one incident, so read it as an example, not a result.

### What are Capy's weak spots?
- Once it ruled out the right cause too early. It tested the checkpoint hypothesis while the first slow save was still in flight, saw nothing, and later ran out of budget. It escalated rather than guess.
- It sometimes names the wrong component on the right node. It once called a PCIe downtraining a slow NIC; the fix still worked.
- It costs about 8× the inference of a single agent and closes a little slower: a median of 29 minutes vs 27.
- When the skeptic model doesn't answer, Capy currently falls back to the deterministic gates instead of failing closed (see [What if a model is down?](#what-if-a-model-is-down)).

## Crusoe and Nemotron

### How do you use Crusoe?
Every agent call goes through Crusoe Managed Inference: one OpenAI-compatible endpoint, four models. We use tool calling, JSON mode, the per-call `enable_thinking` switch and the prefix cache; 55% of Capy's prompt tokens were cache hits. On the product side, every tool maps onto a Crusoe surface: Command Center metrics and logs, Instance Health, VM diagnostics and support tickets, AutoClusters / CMK node replacement, and audit logs. The README has the full table.

### Why Nemotron?
1. **Right-sized models for each job.** Super runs the long tool-calling investigation. Nano, a 30B MoE with 3B active parameters, runs four cheap sweeps in parallel. Lightning writes the case files and runbooks quickly.
2. **Open weights.** A customer can run the whole team on a Nemotron deployment inside their own Crusoe project, so cluster telemetry never leaves it.
3. **A path to getting cheaper.** Verified incidents become fine-tuning data for Lightning, so routine triage can move to the small model.

### What did you learn about the Nemotron models?
- **Super** was a dependable tool-caller across investigations of 10–30 calls. With thinking switched off, it returned fast, well-formed JSON for the contract.
- **Nano** was the right size for parallel triage.
- **Lightning** sometimes returned a 503 "capacity unavailable", so the client falls back to Nano.
- **Qwen 3.8** reasons at length, so the skeptic gets a large token budget and a forced, thinking-off final verdict.

### If you're going for the Nemotron prize, why is the skeptic Qwen?
A skeptic from the same model family tends to share the proposer's blind spots, so falsification works better across families. It's the only non-Nemotron role; every other agent runs on Nemotron.

### What does it cost?
On average, 5.2¢ of inference per incident, against 0.7¢ for the single agent. The extra 4.5¢ buys about a minute of one H100 at typical rental prices, while an incident puts about 100 GPU-hours at stake.

### What would you need from Crusoe to ship this?
Most of it exists today:
- Command Center's Prometheus-compatible metrics and LogsQL logs
- Instance Health
- VM diagnostics
- AutoClusters node replacement
- audit logs

Two additions would make it native. One is an alert hook that triggers Capy; polling the metrics works to start. The other is a least-privilege role that permits Capy's short list of actions and nothing else.

## Safety

### What stops Capy from doing something destructive?
- Tools are read-only by default.
- Actions come from a short whitelist with schemas, and the plan is reduced to known actions and arguments before anything runs.
- Every fix has to pass the proof gate and the skeptic first.
- Disruptive actions (drain, restart, kill, pause) need a human's approval. The ping carries the proof, the skeptic's verdict and the GPU-hour cost.

On the test set it made zero wasted disruptive actions.

### What if nobody answers the ping?
In live mode Capy waits up to 15 minutes. No answer counts as a no: it doesn't act, and it escalates with the case file. In the benchmark, a simulated human approves after 3 minutes.

### What if a model is down?
The client retries 429s and 503s with backoff and falls back per role: Lightning and Qwen both fall back to Nano. If the lead still can't be reached, the run stops with an error and never claims done, so an outage can't produce a false "done".

One honest gap: if the skeptic returns no verdict, Capy currently falls back to the proof gate and verification and notes it in the case file. That happened in one scored run, when our account hit its billing cutoff mid-run; the fix still verified. Escalating instead (failing closed) is the stricter policy, and it's a small change.

### Can logs inject instructions into the agents?
The agents read logs as untrusted text, and text alone can't trigger an action. The plan is reduced to whitelisted actions, the proof gate has to pass on telemetry, the skeptic reviews it, and anything disruptive needs a human. We haven't red-teamed log injection specifically; that comes before any real deployment.

## How it works

### Is this running on a real cluster?
No. A simulator stands in for a 512× H100 Crusoe cluster. You can't inject faults into someone's real 512-GPU job at a hackathon, and scoring "done" needs ground truth. The agents only see the simulator through the tool interface, and that interface is where Command Center's APIs plug in.

### How realistic is the simulator?
- **Timing:** it models synchronous training. Each step takes as long as the slowest rank's compute, plus exposed communication, data wait and overhead, with a checkpoint every 100 steps.
- **Fault signatures:** faults show up the way they do on real hardware:
  - NVML throttle-reason bits and clock drops
  - `nvidia-smi` power-limit messages
  - kernel XID 79, AER and PCIe downtraining lines
  - mlx5 link flaps and NCCL retries
  - `NCCL_IB_HCA` effects
  - slow checkpoint writes
- **Remediation costs:** remediations cost what they would on a real cluster:
  - a drain swaps in a hot spare and rolls back to the last checkpoint (7.5 minutes)
  - a restart takes 3.5 minutes
  - approvals take 3 minutes
  - every tool and model call costs simulated time while 512 GPUs run slow

### How does the proof gate work?
The flight recorder gives per-rank timings for recent steps. To test a claimed cause, Capy replaces that cause's contribution with its healthy value:
- that GPU's compute becomes the fleet median
- exposed comm drops to its 7-day p50
- data wait goes to 0
- checkpoint saves go back to their normal 4 s

It then recomputes the critical path and measures how much of the slowdown disappears. Below 90%, the case stays open.

### Couldn't the lead game the gate by blaming everything?
The gate checks that the claimed causes are enough to explain the slowdown, not that each one is needed. Minimality comes from three other places:
- Each cause has to map to a concrete fix.
- The skeptic is asked whether a cheaper fix exists.
- An over-broad claim means a bigger disruptive fix, which a human sees on the ping, cost included.

### What's the shared brain, and does it help?
After a verified fix, the archivist writes a short runbook: symptom signature, proof, fix. Later incidents get those runbooks in the lead's prompt. We built it and seeded it from dev incidents. We didn't get to measure its effect on the test set before our Crusoe account hit its billing cutoff, so we make no claim yet. The experiment is one flag: `--brain brain/runbooks.jsonl`.

### How long does an incident take?
The median incident is verified fixed 29 minutes after the page, in simulated time. About 15 of those minutes are the 200-step verification itself. The human's approval takes 3 more, and a restart or drain 3.5–7.5. A live run takes 2–8 minutes of wall-clock time, mostly model latency.

## Demo and logistics

### Can we see it live?
Yes, with a Crusoe Managed Inference key. In the console, pick an incident and press "Run live on Crusoe": the agents run on Crusoe in real time, and you approve the ping yourself. Replays of recorded runs, which are the default view and what the video shows, need no key.

### What did you build during the hackathon?
Everything in this repo: the simulator, the agent team, the benchmark, the console, the video pipeline and the diagrams.

### Why didn't you use You.com search?
Incidents are answered from the job's own telemetry. Sending cluster details to a web search would add latency and leak data. A good future use is looking up an unfamiliar XID code or driver erratum, with the query scrubbed.

### What's next?
1. Point the read-only tools at Command Center on a real job.
2. Run a bigger benchmark, with more seeds and more two-fault incidents.
3. Measure what the shared brain adds.
4. Fine-tune Nemotron 3.5 Lightning on verified incidents.
5. Send the ping to Slack or PagerDuty.
6. Make the skeptic fail closed.
