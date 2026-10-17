## Overall

| metric | baseline | capy |
|---|---|---|
| incidents | 34 | 34 |
| fixed (%) | 97.1 | 97.1 |
| correct root cause (%) | 97.1 | 94.1 |
| claimed done (%) | 100.0 | 97.1 |
| false 'done' claims | 1 | 0 |
| false 'done' (% of claims) | 2.9 | 0.0 |
| escalated / gave up (%) | 0.0 | 2.9 |
| GPU-hours lost / incident | 100.65 | 101.603 |
| wasted disruptive actions | 1 | 0 |
| human pings / incident | 0.735 | 0.765 |
| minutes to done (median) | 27.2 | 28.9 |
| tool calls / incident | 12.647 | 27.824 |
| inference $ / incident | 0.0066 | 0.0521 |

## By category

**seen single faults**: baseline: fixed 100.0%, correct 100.0%, false-done 0, GPU-h lost 93.975; capy: fixed 100.0%, correct 100.0%, false-done 0, GPU-h lost 100.446
**held-out fault types**: baseline: fixed 88.9%, correct 88.9%, false-done 1, GPU-h lost 113.8; capy: fixed 88.9%, correct 77.8%, false-done 0, GPU-h lost 110.411
**two-fault incidents**: baseline: fixed 100.0%, correct 100.0%, false-done 0, GPU-h lost 142.5; capy: fixed 100.0%, correct 100.0%, false-done 0, GPU-h lost 50.1

## By fault

| fault | baseline fixed / correct / GPU-h | capy fixed / correct / GPU-h |
|---|---|---|
| checkpoint_stall | 75.0% / 75.0% / 202.025 | 75.0% / 75.0% / 187.5 |
| cpu_contention | 100.0% / 100.0% / 10.5 | 100.0% / 100.0% / 13.0 |
| dataloader_starvation | 100.0% / 100.0% / 13.05 | 100.0% / 100.0% / 16.2 |
| ib_flap | 100.0% / 100.0% / 92.425 | 100.0% / 100.0% / 111.05 |
| nccl_regression | 100.0% / 100.0% / 176.1 | 100.0% / 100.0% / 169.775 |
| pcie_downgrade | 100.0% / 100.0% / 92.3 | 100.0% / 50.0% / 102.35 |
| power_cap | 100.0% / 100.0% / 18.825 | 100.0% / 100.0% / 17.575 |
| power_cap+dataloader_starvation | 100.0% / 100.0% / 142.5 | 100.0% / 100.0% / 50.1 |
| thermal | 100.0% / 100.0% / 106.425 | 100.0% / 100.0% / 116.375 |
| xid79 | 100.0% / 100.0% / 157.025 | 100.0% / 100.0% / 171.7 |

## Not scored (unfinished runs, excluded for every agent)

- cpu_contention-302: capy: run crashed (harness bug), no result
- pcie_downgrade-303: capy: skeptic unavailable: skeptic (nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B) failed: HTTP 402: {"errors":["Organization has no valid payment method"],"id":"851055
- pcie_downgrade-304: capy: lead (nvidia/NVIDIA-Nemotron-3.5-Super-VL) failed: HTTP 402: {"errors":["Organization has no valid payment method"],"id":"19c97935be58b87b80440bbe28934938
- thermal+nccl_regression-402: capy: skeptic unavailable: skeptic (nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B) failed: HTTP 402: {"errors":["Organization has no valid payment method"],"id":"268b0c
- ib_flap+power_cap-403: capy: run crashed (harness bug), no result
- cpu_contention+thermal-404: capy: run crashed (harness bug), no result
