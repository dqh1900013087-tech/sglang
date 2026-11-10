# C128 prefetch alignment: NPU + Mooncake reproduction

## Result

The non-aligned C128 publication and subsequent match acceptance described in
`../../C128_PREFETCH_ALIGNMENT_DEBUG.md` were reproduced on 2026-11-10.
The 6017-token request restored 6016 tokens (47 ordinary pages), but loaded only
2 C128 objects covering 4096 tokens. It published those objects on node 109 at
6016 tokens, accepted that node during matching, and then hit a scheduler
assertion. This is an observed service failure, not an output-accuracy result.

| Candidate range (tokens) | HTTP input tokens | Query pages | Hit pages | Outcome |
| ---: | ---: | ---: | ---: | --- |
| 4096 | 4097 | 32 | 32 | Aligned publication, HTTP 200, storage=4096 |
| 4224 | 4225 | 33 | 32 | Truncated to 4096, aligned publication, HTTP 200 |
| 4480 | 4481 | 35 | 32 | Truncated to 4096, aligned publication, HTTP 200 |
| 6016 | 6017 | 47 | 47 | Non-aligned publication, match accepted, scheduler assertion |

The shorter results do not prove C128 actively enforces alignment. Its
BLOCK_GROUPS verdict does not constrain the overall query hit range in this
baseline. The logs do not identify which individual non-C128 pool constrained
the 33/35-page queries to 32 pages.

## Code and deployment

- Tested source: `94ece9916754a44c732ec70df0ea2c0d33ef44b2`, branch
  `debug-c128-prefetch-alignment`, loaded from `/workspace/sglang/python`.
- The test additionally used `torch.distributed.barrier(group=tp_group.cpu_group)`
  before HCCL prewarm in `Scheduler.maybe_init_hccl_dp_prewarm`.
  **The CPU barrier source change is excluded from this evidence commit.**
  It remains an uncommitted local change on the test machine.
- DeepSeek-V4-Flash, FP8 checkpoint, Ascend950DT, physical devices 4,5,6,7.
- TP=4, attention DP=4, ordinary DP=1, requests pinned to routed_dp_rank=0.
- MTP disabled; page_size=128; C128 block_pages=16; group_tokens=2048.
- Host mode `cache`, write policy `write_through`, prefetch `wait_complete`.
- Launch script requested chunked prefill=512; effective server value was 128.
- MemFabric 1.3.1, Mooncake 0.3.12.post1, device_urma via TENT, store protocol tcp.
- `SGLANG_DEBUG_C128_PREFETCH=1`, log level info.
- Installed the repository-required `deepseek-recipe==0.2.0` after the initial
  startup attempt failed due to the missing package.
- Full effective configuration is in `server_info.json`. The installed package
  version string differs from the checked-out source; use the source commit above.
- Timestamps in the raw logs use the container clock. The directory name and
  log timestamps are preserved without conversion.

## Request sequence

1. Flush local caches while idle. Current reset code does not delete Mooncake
   objects. Keep the same model, storage identity and cache salt within the run.
2. Seed a deterministic 8193-token prefix with one generated token, temperature=0,
   ignore_eos=True. Token IDs are `1000 + (i * 37) % 90000`.
3. Observe Mooncake put-completion counter growth (220 objects) and five stable
   follow-up samples, two seconds apart. See `seed_long_backup_metrics.json`.
4. Before each candidate, flush local caches and require HTTP 200; send prefixes
   of 4097, 4225, 4481, and 6017 tokens in that order, sharing the seed cache salt.
5. The first three candidates returned HTTP 200. The fourth caused scheduler
   termination. Its request and flush response are saved, but it has no successful
   HTTP generation response. Server logs prove actual L3 transfer and publication.

Each request has a unique request ID. The exact JSON requests and responses are
included. `probe.py` is the original script used, including the later planned
cases; those cases did not run after the 6016-token failure. Its `results.json`
contains only requests that returned successfully. `summary.json` and
`root_6016_failure.log` also cover the failed request.

## Decisive evidence

From DP0/TP0, request `root_6016-b1ae4871` (selected fields):

```text
phase=query storage_start=0 query_pages=47 kv_hit_pages=47 end_mod_group=1920 raw_c128_query_verdict=2
phase=result completed_tokens=6016 planned_c128_keys=2 loaded_c128_objects=2 all_succeeded=True
phase=insert_host inserted_host_node=109 completed_tokens=6016 dropped=False
phase=publish target_id=109 target_tokens=6016 end_mod_group=1920 published_objects=2 unaligned=True
phase=match_validator node_id=109 node_tokens=6016 end_mod_group=1920 accepted=True
AssertionError: req.kv.cache_protected_len=6016, len(new_indices)=4096, page_aligned_len=4096
```

`root_6016_failure.log` preserves the original lines and complete first assertion
stack. `server.log` is the complete server log for the run with the barrier.
`master.log` is a byte-for-byte snapshot prefix; the still-running master can
continue appending to its original log after this capture.

## Scope and limits

This run proves non-aligned publication and acceptance of the same node, followed
by a scheduler assertion. It does not verify a later query starting from a
non-aligned position: the current request already caused the service to exit.
The 6144 control, retained-prefix cases and separately seeded non-aligned cases
were not executed. No output-accuracy comparison was performed.

The CPU barrier allowed this startup to complete; one successful startup is not
proof that all possible communication initialization races are fixed. Without
it, an earlier attempt stalled: DP0/2/3 were in HCCL synchronize while DP1 was in
Mooncake warmup. See `server_stalled_warmup.log` and `stack_12540.txt` through
`stack_12543.txt`. `server_failed_missing_recipe.log` is the earlier, separate
missing-dependency failure. `server_retry_before_barrier.log` is the retry
interrupted before applying the barrier; do not count it as another completed test.

At capture end, SGLang had exited and port 8001 refused connections; Mooncake
master remained running. These are historical observations, not a live status.

## Re-run

The following uses the test environment's existing model and Ascend installation.
Review the saved launch script and the startup-barrier prerequisite above first.
Ensure port 8001 is available. Keep the diagnostic environment variable enabled
in every worker. The launch script starts the service; the probe flushes its
local caches and can reproduce the scheduler crash. It does not patch SGLang.

```bash
EVIDENCE=/workspace/sglang/c128_prefetch_alignment_results/20261110_070605
RUN=/home/dqh/c128_validation/repro_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN"
SGLANG_DEBUG_C128_PREFETCH=1 LOG_DIR="$RUN" \
  SGLANG_EXTRA_ARGS="--hicache-host-memory-mode cache --log-level info" \
  bash "$EVIDENCE/run_mooncake.sh"
python3 -u "$EVIDENCE/probe.py" "$RUN" > "$RUN/probe.log" 2>&1
```

The original complete run resides at
`/home/dqh/c128_validation/20261110_070605` on the test machine.
`MANIFEST.sha256` checks the committed evidence files (excluding the manifest itself).
