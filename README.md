# Qwen 27B on Thor: DFlash2 study

Measured on NVIDIA IGX Thor, ARM64 / SM110, 25–26 October 2026. This repository contains scripts, stored synthetic prompts and outputs, timing evidence, serving configurations, and a Chisel draft-head adaptation. No model weights are included.

**The best measured mixed score was 38.65 tokens/s:** QUASAR Qwen3.8-27B NVFP4 with the Chisel NVFP4A16 DFlash2 drafter and seven speculative tokens. Draft-head FP8 and vocabulary reduction did not improve that score. This is a small integration study of published techniques, not a new decoding algorithm or a general model leaderboard.

| Configuration | Mixed request tok/s | Prose decode | Code decode | Long-input decode |
|---|---:|---:|---:|---:|
| Earlier MTP3, FP8 target head, 64K draft vocabulary | 34.53 | 32.47 | 42.81 | 35.64 |
| DFlash2 NVFP4A16, 7 drafts, full BF16 head | **38.65** | 32.51 | 73.19 | 37.24 |
| Same, full FP8 draft head | 37.10 | 30.48 | 74.94 | 35.09 |
| Same, 100,032-token FP8 draft head | 37.08 | 32.90 | 74.29 | 32.23 |
| Same, 100,032-token BF16 draft head | 35.20 | 29.74 | 70.26 | 31.79 |
| DFlash2 NVFP4A16, 10 drafts | 33.50 | 28.17 | 85.41 | 27.82 |
| DFlash2 BF16, 7 drafts | 32.29 | 29.59 | 65.20 | 26.61 |

All rates are tokens/s. Mixed request rate includes first-token latency. Decode rate excludes the first token batch. These measures are different. The seven-token result is 11.9% above the earlier MTP result for these prompts. The ten-token result has the highest code decode rate but a lower mixed score.

## Read and check the evidence

- [Method and limits](docs/method.md): workload, checks, failed starts, memory limits, and what remains untested.
- [Reproduction guide](docs/reproduction.md): pinned sources, build, configuration templates, and benchmark scripts.
- [Upstream credits](NOTICE.md): authors, source links, licenses, and local changes.
- [All measured rows](results/comparison.json), [head checks](results/head-audit.json), and [version pins](versions.json).
- `results/<variant>/`: synthetic prompts, outputs, exact token batches, relative timing, focused quality checks, and filtered counter snapshots.

Check the stored results with Python 3. This command does not contact a server or execute generated code:

```sh
python3 scripts/audit_results.py
```

## Main limits

One request at a time, 4K context, three synthetic prompt types, three rotated rounds, and 256 forced output tokens per timing request. Controls were measured earlier, not rerun as paired trials. There are no confidence intervals or broad quality scores. Longer context, concurrent load, multilingual use, SGLang, and Chisel GDN replay were not tested.

The MTP and DFlash2 rows compare complete serving configurations, with different cache sizes and head settings. They do not isolate the decoding algorithm. The full/BF16 head control is also an earlier measurement. Small rate differences must not be treated as a firm ranking on other workloads.

A failed model load caused a host-wide out-of-memory event despite high `MemAvailable`. The successful trials do not establish safe coexistence with other services. Private service controls, host logs, GPU identifiers, addresses, and machine paths are excluded. No new benchmark was run to prepare this release.

Code is Apache-2.0 unless otherwise stated. Bundled upstream files retain their licenses. Model weights and runtime dependencies have their own terms. See [NOTICE.md](NOTICE.md).
