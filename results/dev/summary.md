## Overall

| metric | baseline | capy |
|---|---|---|
| incidents | 18 | 18 |
| fixed (%) | 100.0 | 94.4 |
| correct root cause (%) | 94.4 | 100.0 |
| claimed done (%) | 100.0 | 94.4 |
| false 'done' claims | 0 | 0 |
| false 'done' (% of claims) | 0.0 | 0.0 |
| escalated / gave up (%) | 0.0 | 5.6 |
| GPU-hours lost / incident | 103.639 | 111.289 |
| wasted disruptive actions | 0 | 1 |
| human pings / incident | 0.722 | 0.833 |
| minutes to done (median) | 27.0 | 29.4 |
| tool calls / incident | 10.056 | 28.667 |
| inference $ / incident | 0.0042 | 0.0548 |

## By category

**seen single faults**: baseline: fixed 100.0%, correct 94.4%, false-done 0, GPU-h lost 103.639; capy: fixed 94.4%, correct 100.0%, false-done 0, GPU-h lost 111.289
**held-out fault types**: 
**two-fault incidents**: 

## By fault

| fault | baseline fixed / correct / GPU-h | capy fixed / correct / GPU-h |
|---|---|---|
| dataloader_starvation | 100.0% / 100.0% / 12.233 | 100.0% / 100.0% / 16.233 |
| ib_flap | 100.0% / 100.0% / 121.767 | 100.0% / 100.0% / 136.5 |
| nccl_regression | 100.0% / 100.0% / 155.367 | 100.0% / 100.0% / 183.133 |
| power_cap | 100.0% / 100.0% / 54.467 | 66.7% / 100.0% / 36.567 |
| thermal | 100.0% / 66.7% / 112.5 | 100.0% / 100.0% / 110.667 |
| xid79 | 100.0% / 100.0% / 165.5 | 100.0% / 100.0% / 184.633 |
