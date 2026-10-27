# Routed NVFP4 MoE

Kernels for the Open Kernel Challenge problem [`muse_moe_w4a16`](https://openkernelchallenge.dev/problems/muse_moe_w4a16). Team **Aravind-11**.

## Board result

Numbers are from the public leaderboard on 27 October 2026. The judge is an H200. The baseline is SGLang, geomean 13448.5 µs. My best scored run is **1.629×** (8254 µs), confidence interval [1.605, 1.654].

Rank **493 of 936**. 

| Shape | SGLang (µs) | Ours (µs) | Speedup |
|---|---:|---:|---:|
| decode-1 | 77.5 | 47.5 | 1.630× |
| decode-16 | 188.4 | 120.3 | 1.566× |
| decode-32 | 206.3 | 154.8 | 1.333× |
| verify-8 | 150.5 | 96.8 | 1.556× |
| verify-128 | 467.8 | 246.7 | 1.896× |
| verify-256 | 610.8 | 327.6 | 1.865× |

Leaderboard: https://openkernelchallenge.dev/leaderboard

## What is in this repo

- `TASK.md`, `reference.py`, `checks.py`, `baseline.py`, `nvfp4.py` — the problem, the reference, and the SGLang baseline.
- `submission_v5.py`, `submission_v6.py` — the early kernels described above.

There is no `submission.py`. These files are not the current entry as the entries progress over time but will keep the files updated as much as I could
