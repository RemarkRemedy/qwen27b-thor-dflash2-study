# Method and limits

## Workload

Each configuration had one warmup, then three rounds of short prose, Python code, and station-record prose. Each round rotated the case order. The long speed input had 3,008 plain-text tokens before the chat template. Each timing request used 256 output tokens, `ignore_eos=true`, greedy selection, thinking disabled, streaming, and exact returned token IDs. Context and batched-token limits were 4,096; maximum concurrent sequences was one.

The primary score is the sum of output tokens divided by the sum of full request durations across nine requests. Per-case decode values are medians of three runs. Decode rate counts tokens after the first returned token batch and divides by the interval from the first to last token batch. It does not count a whole speculative batch as one token. Request rates and first-token latency remain in the raw files.

Acceptance counters use the difference between the warmup and final `code_r3` snapshots. Only the three speculative-decoding counters are exported; deployment labels and unrelated metrics are omitted. These counters cover the nine speed requests, not later quality requests.

## Runtime

The target is QUASAR-QAT Qwen3.8-27B NVFP4. Mjolnir provides target full attention, with FP8 HND KV cache and FlashInfer prefill. Target linear operations selected `FlashInferCutlassNvFp4LinearKernel`. The native DFlash2 V2 path uses FlashInfer for the drafter's sliding-window attention and unquantized draft KV. The NVFP4A16 drafter selected `MarlinNvFp4LinearKernel`. Fused CUDA GDN remained active; Chisel GDN replay was not enabled.

Seven-token DFlash2 used 3 GiB KV cache; ten tokens used 4 GiB. The earlier MTP3 control used 2 GiB, an FP8 target head, and a 65,536-token draft vocabulary. The container memory cap was 48 GiB. These are configuration comparisons, not comparisons at equal memory use. Logged model allocations were 21.53 GiB for the BF16 drafter and 19.13 GiB for the original NVFP4 drafter setup. These figures exclude some runtime and host memory.

## Draft-head adaptation

Chisel's upstream reduced-vocabulary code preserves an already-FP8 target head. This target's head is BF16. The local adapter adds draft-only conversion with per-channel FP8 weights and dynamic per-token FP8 activations, using vLLM's `CompressedTensorsW8A8Fp8` implementation. Runtime selected `CutlassFP8ScaledMMLinearKernel` on SM110.

The adapter checks the target head's tensor identity, pointer, sampled values, shape, and BF16 dtype. It checks all vocabulary mappings and actual top-k candidate IDs and values. The target head remains full size: 248,320 by 5,120. The pinned vLLM candidate path requires the Chisel mapping. Newer versions must be checked for an existing mapping to avoid applying it twice.

Eight seeded random activation rows gave relative L2 error 0.037226 for full FP8 and 0.037082 for reduced FP8, below the adapter's 0.05 threshold. This is a numerical check, not a language-quality score. Each adapted configuration creates a separate draft head. It does not free the shared target head. Loaded model allocations were 20.09 GiB for reduced BF16, 20.33 GiB for full FP8, and 19.62 GiB for reduced FP8. Thus smaller draft weights do not imply less total memory than the shared-head baseline.

## Validation and failures

All seven published rows have nine timing requests with no recorded guard pause, six factual/conversation/repeated-prefix checks, a natural-stop code response, and 13 stored CPU code-check results. The code checks cover Fibonacci output and invalid input types/values. They do not prove broad model quality. Quality-client decode estimates use text chunks; only the speed-client token-ID rates enter the performance table.

The supplied Mjolnir reference harness was used during the original work for new page/query/layout shapes at lengths 128 and 4,096. Previously passed identical shapes were reused. Private harness logs are not in this export, so this statement is an experiment note, not an independently reproducible proof from the exported JSON.

The six completed DFlash2 configurations are published with the earlier MTP control. Failed startup attempts have no throughput score:

1. A 2 GiB DFlash2 cache was rejected: one 4K request required about 2.62 GiB. Seven-token tests then used 3 GiB.
2. One later load caused a real host-wide OOM; an unrelated helper process was killed and restarted. The test container did not report a container OOM. High `MemAvailable` and a 48 GiB container cap did not prevent the host event.
3. Releasing only test-checkpoint file-cache pages with `posix_fadvise(DONTNEED)`, including after loading repopulated the cache, preceded successful runs without a new observed OOM kill. This sequence does not prove that the cache operation alone guarantees safety.
4. Two attempts stopped at unrelated service-recovery boundaries. A service guard timer was corrected in the private host integration. That integration is outside this portable export.

No host lifecycle controller or production deployment is supplied. The portable command templates do not include the original checkpoint-cache release or service coordination. They are not evidence that a raw Docker launch is safe on a shared host.

## Open questions

Only three prompt types and three repeats per type were measured. Repeated prompts can benefit from prefix caching. Temperature-zero checks do not establish the full sampling distribution. There was no broad accuracy evaluation, long-context trial, concurrent-load trial, or controlled power study. Draft length and cache size both changed in the ten-token comparison. Prior controls were not paired with each follow-up. These limits prevent a universal speed or quality claim.
