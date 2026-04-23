# GEMM + Reduce-Scatter: Implementation Progress

**Last updated:** 2026-04-22  
**Owner:** pkousha@nvidia.com  
**Issue:** https://github.com/flashinfer-ai/flashinfer/issues/2435  
**Reference PR (all_gather_matmul):** https://github.com/flashinfer-ai/flashinfer/pull/2665

> **For any new session:** read this file first to restore context before touching any code.
> Cross-reference with `design.md` (algorithm) and `comparison.md` (benchmarks).

---

## Files Created

```
flashinfer/comm/gemm_reduce_scatter/
├── PROGRESS.md          ← this file
├── design.md            ← full algorithm design, cache analysis, mbarrier, test report
├── comparison.md        ← benchmark results, baseline rationale, findings
├── configs.py           ← CACHE_LINE_WORDS=32, SIGNAL_DTYPE (BARRIER_MODE removed)
├── gemm_reduce_scatter_triton.py  ← Triton GEMM+Push kernels (original design)
├── gemm_reduce_scatter_cublas.py  ← cuBLAS CE loop backend (new, 2026-04-22)
├── gemm_reduce_scatter.py         ← dispatcher: backend param + FLASHINFER_GRS_BACKEND env
└── __init__.py                    ← exports all three public functions

flashinfer/comm/__init__.py          ← gemm_reduce_scatter exported
tests/comm/test_gemm_reduce_scatter.py  ← correctness + benchmark + CLI
```

---

## Algorithm Summary

Three kernels, two concurrent streams, Design B (chunk-outer, dest-inner loop):

```
Kernel 1: barrier        — one-time global sync (main_stream)
Kernel 2: GEMM+Push      — compute_stream; Triton GEMM tiles written to peer
                           recv_buf via NVLink; inter-CTA arrive barrier per
                           (chunk, dest) pair signals destination rank
Kernel 3: Wait+Reduce    — reduce_stream; spin-waits on rs_signal[src,chunk],
                           accumulates recv_buf[src] into output
```

Full description: `design.md §3–8`

---

## Test Results

### Correctness

| Environment | GPUs | ws=2 | ws=4 | ws=8 |
|---|---|---|---|---|
| L40S SM89 PCIe | 4× L40S | ✓ 4/4 | ✓ 4/4 | — |
| H100 SM90 NVLink | 8× H100 NVL | ✓ 4/4 | ✓ 4/4 | ✗ GPUs 2,3,7 in recovery |

8 correctness cases: M=4096/8192/16384 (bfloat16) + M=4096 (float16), ws=2 and ws=4.

Test invocation (non-standard — see P1 item):
```bash
cd /home/pkousha/projects/flashinfer
PYTHONPATH=/storage/pkousha/projects/flashinfer \
CUDA_VISIBLE_DEVICES=0,1,4,5 \
.venv/bin/pytest tests/comm/test_gemm_reduce_scatter.py::test_correctness \
    -v --timeout=600 --import-mode=prepend
```

### Benchmarks (H100 SM90, ws=4, K=8192, N=2048, bfloat16)

All four compute `(Σ X_local @ W_local) / world_size`:

| M | naive | fused_matmul_rs | our spin | our mbarrier |
|---|---|---|---|---|
| 8192 | 1.2ms | 23.1ms | 77.0ms | 77.4ms |
| 16384 | 2.3ms | 47.9ms | 57.1ms | 64.3ms |
| 65536 | **9.3ms** | 194.9ms | 127ms | **124ms** |

Key finding: **naive (cuBLAS + NCCL) wins everywhere**. Our Triton GEMM is 10-15×
slower than cuBLAS. We beat `fused_matmul_reduce_scatter` at M=65536 (1.56×) but
lose to naive. See `comparison.md` for full analysis and root causes.

---

## What's Done

- [x] Algorithm design (Design B chunk-outer, dest-inner)
- [x] Cache-line padding for arrive_count (Fix #1: ARRIVE_STRIDE=32)
- [x] Cache-line padding for rs_signal_buf (Fix #2: SIGNAL_STRIDE=32)
- [x] Separate symm_mem tensors for barrier/data/signals (no offset aliasing)
- [x] Barrier kernel matching AG+GEMM pattern (grid=1, static_range)
- [x] Inter-CTA arrive barrier (unique counter slot per (chunk,dest), no reset)
- [x] mbarrier variant (ld.global.acquire.sys.b32 on SM≥90)
- [x] BARRIER_MODE env var dispatch (in Triton backend only)
- [x] cuBLAS CE loop backend (`gemm_reduce_scatter_cublas.py`)
- [x] Two-backend dispatcher with `backend` param + `FLASHINFER_GRS_BACKEND` env
- [x] Fixed double-buffering race (world_size buffers, not 2)
- [x] Fixed BARRIER_MODE dead code (configs.py now reads FLASHINFER_GRS_BARRIER)
- [x] Fixed Wait+Reduce launch order (before CE loop, not after)
- [x] cuBLAS backend test coverage added (backend parametrised in test_correctness/test_benchmark)
- [x] GemmReduceScatterWorkspace class (gemm_reduce_scatter_workspace.py)
- [x] Both backends accept optional `workspace=` parameter (backward-compatible)
- [x] Benchmark script with workspace: benchmarks/gemm_reduce_scatter/bench_workspace.py
- [x] Benchmark results saved to: benchmarks/gemm_reduce_scatter/results/
- [x] Correctness tests passing on L40S and H100
- [x] Benchmark vs naive + fused_matmul_reduce_scatter
- [x] design.md, comparison.md, this PROGRESS.md

---

## What's Left (Prioritised)

### P0 — Blockers

**1. Replace Triton GEMM with cuBLAS**  
The most critical gap. Our Triton GEMM is 10-15× slower than `torch.mm` (cuBLAS).
Naive (`torch.mm + NCCL reduce_scatter`) beats our kernel everywhere.  
Fix: use `torch.mm` for the GEMM (or call `X_local @ W_local` via cuBLAS), then
use our push-wait communication infrastructure for the scatter+reduce.
This changes the architecture — the GEMM no longer writes tiles directly to peer
buffers; instead we complete the GEMM first, then scatter via CE and reduce.

**2. BLOCK_N=128 register pressure — validate before changing**  
A subagent computed ~274 theoretical registers/thread for `_wait_reduce_kernel`
(128 acc + 128 partial_fp32 + ~18 overhead) vs SM90 limit of 255.  
**This is a hypothesis, not a confirmed bug.** No PTX inspection, profiler counter,
or `st.local`/`ld.local` spill evidence exists in-tree.  ptxas can often optimize
register allocation below the theoretical upper bound.  
Additionally, `_BLOCK_N` is shared between GEMM+Push and Wait+Reduce — reducing
it would halve GEMM tile width and could hurt GEMM throughput.  
**Action:** validate with `TRITON_PRINT_AUTOTUNING` or Nsight Compute register
pressure counters before changing defaults.  If spill is confirmed, evaluate
separate BLOCK_N values for the two kernels rather than a global change.

**3. ~~Workspace class~~ DONE** — `GemmReduceScatterWorkspace` implemented.
Pre-allocates all symm_mem tensors, streams, remote pointers, staging buffers
once in `__init__`. Both backends accept optional `workspace=` parameter.

### P1 — Important

**4. Fix design.md: "mbarrier is an SM90 primitive" is wrong**  
Basic mbarrier exists from SM80 (Ampere). Cluster-scoped ops require SM90.  
Line 332: change to "mbarrier basic operations exist from SM80; cluster-scoped
operations require SM90."

**5. Rename mbarrier env var or document accurately**  
`FLASHINFER_GRS_BARRIER=mbarrier` uses `ld.global.acquire.sys` not true PTX
mbarrier (no warp parking). Benchmarks show negligible benefit (127ms vs 124ms).
Option A: rename to `acquire_sys`. Option B: keep name, add prominent caveat.

**6. Remove acquire.sys performance claim in design.md**  
design.md §8 claims reduced L2 pressure. Benchmarks showed 2% difference (within
noise). Remove or soften to "marginal/unvalidated at current workloads."

**7. Add occupancy guard before launch**  
Assert `NUM_SMS_compute <= NUM_SMS` at Python launch time. The inter-CTA barrier
assumes all CTAs are resident simultaneously; this only holds when
grid ≤ actual SM count. Currently this is guaranteed by construction but should
be explicit.

**8. Remove `enable_symm_mem_for_group` deprecation in tests**  
`tests/comm/test_gemm_reduce_scatter.py:117` calls the deprecated function,
printing FutureWarning on every run. Remove it (PyTorch 2.11+ no-op).

**9. Fix test invocation (non-standard)**  
Tests require `--import-mode=prepend` + `PYTHONPATH` due to `pytest.ini` using
`--import-mode=importlib` which breaks `mp.spawn` module resolution.
Either fix in conftest.py or document the required invocation prominently.

**10. Decide avg vs sum in public API**  
`gemm_reduce_scatter()` returns a sum; vLLM's `GEMMReduceScatterPattern` uses avg
(÷ world_size). The benchmark adds `out.div_(world_size)` manually. Decide:
should the public API accept a `reduce_op` parameter, or always return sum?

**11. ws=8 correctness test**  
GPUs 2, 3, 7 on gpuh100x8-01 are in `GPU Recovery Action: Reset` (hardware issue,
requires admin intervention or reboot). Rerun when restored.

### P2 — Next Sprint

**12. cuTile backend (SM≥100, Blackwell)**  
`gemm_reduce_scatter.py` raises `NotImplementedError` for SM≥100. Follow
`all_gather_matmul_cutile.py` pattern from PR #2665.

**13. Tail chunk support**  
Currently asserts `M_local % CHUNK_SIZE_M == 0`. Add masked final chunk +
adjusted barrier expected counts for non-divisible M.

**14. vLLM torch.compile integration**  
Add `GEMMReduceScatterPattern` to vLLM's `AsyncTPPass` replacing
`torch.ops.symm_mem.fused_matmul_reduce_scatter`. Mirror `AllGatherGEMMPattern`.

**15. Profiling-based validation**  
Use Nsight Compute to measure actual L2 traffic, stall reasons, and overlap
efficacy in spin vs mbarrier mode. Validate or disprove design.md §8 claims.

**16. Pin PyTorch version in comparison.md**  
`fused_matmul_reduce_scatter` internals can change across versions. Add tested
PyTorch commit/version (currently 2.11.0+cu130).

### P3 — Backlog

**17. AOT registration** in `flashinfer/aot.py`  
**18. CUDA graph capture testing**  
**19. Auto-tune CHUNK_SIZE_M and SM split** (75/25 is untuned)  
**20. ws=8 benchmark** (needs healthy 8-GPU node)

---

## Key Design Decisions (do not revisit without reason)

| Decision | Rationale |
|---|---|
| Design B (chunk-outer, dest-inner loop) | All dest ranks get chunk-0 signal after one round; maximises overlap vs dest-outer |
| Separate barrier_buf / rs_signal_buf / recv_buf symm_mem tensors | Avoids `get_signal_pad` offset aliasing bug (element vs byte) and 16B alignment issue |
| ARRIVE_STRIDE=SIGNAL_STRIDE=32 | Each counter/signal on its own 128-byte cache line; eliminates false sharing |
| grid=(NUM_SMS_compute,) exactly | Ensures all barrier-participating CTAs are resident simultaneously |
| arrive_count unique slot per (chunk,dest), no reset | Simpler than flip-flop; safe because each slot used exactly once |
| Barrier: grid=(1,), static_range loops | Matches AG+GEMM PR #2665 exactly; multi-block had tl.program_id runtime indexing bug |
| Reference for correctness: fp32 matmul + all_gather | Fairer than NCCL (which uses bf16 ring reduce); atol=3.0 for bf16 quantisation |

---

## Known Bugs / Gotchas

- `_BLOCK_N=128` causes register spill on SM90 (274 > 255 regs/thread). Fix in P0.
- `tl.inline_asm_elementwise` constraint `"=r, l"` — `l` is 64-bit pointer, confirmed from Triton docs.
- Both PTX orderings `ld.global.acquire.sys.b32` and `ld.acquire.sys.global.b32` are valid — ptxas accepts both (confirmed from FlashInfer codebase).
- `symm_mem.get_signal_pad` storage_offset is in **elements**, not bytes.
- SM90 portable cluster max = 8 CTAs; non-portable = 16 (requires `cudaFuncAttributeNonPortableClusterSizeAllowed`).
- Tests require `--import-mode=prepend` + `PYTHONPATH` (see P1 #9).
- GPUs 2, 3, 7 on gpuh100x8-01 are in hardware recovery state as of 2026-04-22.

---

## Session Log

| Date | What happened |
|---|---|
| 2026-04-10 | Design discussions: algorithm, Design B loop ordering, cache footprint, mbarrier |
| 2026-04-17 | Full implementation: all 5 files created, design.md written |
| 2026-04-22 | Chunk-by-chunk review; 7 bugs fixed (barrier grid, test mp.spawn, etc.); correctness tests passing on L40S and H100; benchmarks run; comparison.md written; Codex review fact-checked |
