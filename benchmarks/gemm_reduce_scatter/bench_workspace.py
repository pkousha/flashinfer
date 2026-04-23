"""
Full GEMM+Reduce-Scatter benchmark with workspace pre-allocation.
Backends: naive | triton-spin | triton-mbarrier
Tokens (M): 1K – 64K, world sizes: ws=2, ws=4, ws=8

Results saved to: benchmarks/gemm_reduce_scatter/results/

Phase 1: pre-warm all Triton kernel variants (single GPU).
Phase 2: benchmark with pre-allocated workspace (no per-call rendezvous).

Statistics reported per (M, backend, ws): mean, std, min, max, median, p5, p95.
Each call is timed individually with a CUDA event pair + synchronize, giving
true per-call latency (latency mode, not throughput mode).
"""
import os, sys, socket, multiprocessing, time, json, statistics
from datetime import datetime
from pathlib import Path

REPO = "/storage/pkousha/projects/flashinfer"
RESULTS_DIR = Path(REPO) / "benchmarks" / "gemm_reduce_scatter" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

os.environ["PYTHONPATH"] = REPO
sys.path.insert(0, REPO)

import torch, torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

K       = 8192
N       = 2048
DTYPE   = torch.bfloat16
N_ITERS = 30    # per-call measurements (with sync between calls)
WARMUP  = 5     # warmup calls before measurement
CHUNK   = 1024

# M=131072 excluded: recv_buf=(ws, M_local, N) reaches 512 MB for ws=2,
# making symm_mem.rendezvous slow (~minutes). Test up to 65536 for now.
ALL_M = [1024, 2048, 4096, 8192, 16384, 32768, 65536]

# cublas backend is WIP (NVLink DMA visibility bugs); excluded from benchmarks.
# See flashinfer/comm/gemm_reduce_scatter/CUBLAS_WIP.md for details.
#
# "fused": torch.ops.symm_mem.fused_matmul_reduce_scatter — PyTorch built-in
# kernel, the vLLM production baseline this work aims to replace.
BACKENDS = ["naive", "fused", "spin", "mbarrier"]

WORKER_TIMEOUT_S = 600   # 10 min max per world-size run


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Pre-warm Triton kernels (single GPU, populates ~/.triton/cache)
# ─────────────────────────────────────────────────────────────────────────────

def prewarm():
    from flashinfer.comm.gemm_reduce_scatter.gemm_reduce_scatter_triton import (
        _wait_reduce_kernel, _gemm_push_kernel, _gemm_push_kernel_mbarrier,
        _BLOCK_M, _BLOCK_N, _BLOCK_K, _GROUP_SIZE_M,
        _CHUNK_SIZE_M, _SIGNAL_STRIDE, _ARRIVE_STRIDE,
    )
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    for ws in [2, 4, 8]:
        M_local = max(CHUNK, (4096 // ws // CHUNK) * CHUNK)
        num_chunks = M_local // _CHUNK_SIZE_M
        if num_chunks == 0:
            num_chunks = 1
            M_local = num_chunks * _CHUNK_SIZE_M

        recv   = torch.zeros(ws, M_local, N, device=device, dtype=DTYPE)
        sig    = torch.ones(ws, num_chunks, _SIGNAL_STRIDE, device=device, dtype=torch.uint32)
        out    = torch.empty(M_local, N, device=device, dtype=DTYPE)
        arrive = torch.zeros(num_chunks * ws * _ARRIVE_STRIDE, dtype=torch.int32, device=device)

        t0 = time.time()
        _wait_reduce_kernel[(1,)](recv, sig, out, M_local=M_local, N=N, num_chunks=num_chunks,
            stride_per_src=M_local*N, CHUNK_SIZE_M=_CHUNK_SIZE_M, SIGNAL_STRIDE=_SIGNAL_STRIDE,
            BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N, RANK=0, WORLD_SIZE=ws)
        torch.cuda.synchronize()
        print(f"  ws={ws} _wait_reduce_kernel ... {time.time()-t0:.1f}s", flush=True)

        X = torch.randn(ws * M_local, 256, device=device, dtype=DTYPE)
        W = torch.randn(256, N, device=device, dtype=DTYPE)
        recv_ptrs = tuple(recv[s] for s in range(ws))
        sig_ptrs  = tuple(sig[s] for s in range(ws))
        kw = dict(M_local=M_local, N=N, K_local=256, num_chunks=num_chunks,
                  CHUNK_SIZE_M=_CHUNK_SIZE_M, ARRIVE_STRIDE=_ARRIVE_STRIDE,
                  SIGNAL_STRIDE=_SIGNAL_STRIDE, BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N,
                  BLOCK_K=_BLOCK_K, GROUP_SIZE_M=_GROUP_SIZE_M, RANK=0, WORLD_SIZE=ws)

        t0 = time.time()
        _gemm_push_kernel[(1,)](X, W, recv_ptrs, sig_ptrs, arrive, **kw)
        torch.cuda.synchronize()
        print(f"  ws={ws} _gemm_push_kernel (spin) ... {time.time()-t0:.1f}s", flush=True)

        arrive.zero_()
        t0 = time.time()
        _gemm_push_kernel_mbarrier[(1,)](X, W, recv_ptrs, sig_ptrs, arrive, **kw)
        torch.cuda.synchronize()
        print(f"  ws={ws} _gemm_push_kernel_mbarrier ... {time.time()-t0:.1f}s", flush=True)

    print("Pre-warm complete.\n", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_stats(times_ms: list) -> dict:
    """Compute descriptive statistics from a list of per-call milliseconds."""
    n = len(times_ms)
    s = sorted(times_ms)
    mean  = sum(s) / n
    std   = statistics.stdev(s) if n > 1 else 0.0
    p5    = s[max(0, int(n * 0.05))]
    p25   = s[max(0, int(n * 0.25))]
    med   = s[n // 2]
    p75   = s[min(n - 1, int(n * 0.75))]
    p95   = s[min(n - 1, int(n * 0.95))]
    return {
        "mean": mean, "std": std,
        "min": s[0], "max": s[-1],
        "p5": p5, "p25": p25, "median": med, "p75": p75, "p95": p95,
        "raw": times_ms,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Benchmark worker
# ─────────────────────────────────────────────────────────────────────────────

def worker(world_size, rank, port, result_path, log_path):
    os.environ["PYTHONPATH"] = REPO

    _log = open(log_path, "a") if rank == 0 else None
    def log(msg=""):
        if rank == 0:
            print(msg, flush=True)
            _log.write(msg + "\n"); _log.flush()

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", init_method=f"tcp://localhost:{port}",
                            rank=rank, world_size=world_size)
    group = dist.group.WORLD
    symm_mem.enable_symm_mem_for_group(group.group_name)

    from flashinfer.comm.gemm_reduce_scatter import gemm_reduce_scatter
    from flashinfer.comm.gemm_reduce_scatter import GemmReduceScatterWorkspace
    from flashinfer.comm.gemm_reduce_scatter.configs import Configs
    K_local = K // world_size

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    col_w = 16   # column width for mean(std) display
    if rank == 0:
        hdr  = f"\n{'='*110}\n"
        hdr += f"  ws={world_size}  K={K}  N={N}  bf16  H100 NVL  n_iters={N_ITERS}  warmup={WARMUP}\n"
        hdr += f"  Timing: per-call latency (CUDA events, sync between calls)\n"
        hdr += f"  Backends: naive=torch.mm+NCCL  fused=symm_mem.fused_matmul_reduce_scatter  spin/mbar=our kernel\n"
        hdr += f"{'='*110}\n"
        hdr += f"  {'M':>7}"
        for b in BACKENDS:
            hdr += f"  {b+' mean±std':>{col_w}}"
        if "naive" in BACKENDS:
            for b in [x for x in BACKENDS if x != "naive"]:
                hdr += f"  {b[:5]+'/naiv':>10}"
        hdr += f"\n  {'-'*106}"
        log(hdr)

    # ------------------------------------------------------------------
    # Per-call timing function.
    # All ranks sync via dist.barrier() before each call to ensure they
    # start together (critical for NCCL collectives and spin-barrier kernels).
    # The barrier is outside the CUDA event pair so its latency is not
    # included in the measurement.
    # ------------------------------------------------------------------
    def _time_calls(fn, n):
        """Return list of per-call milliseconds with inter-rank sync."""
        times = []
        for _ in range(n):
            dist.barrier(group)          # all ranks start simultaneously
            torch.cuda.synchronize()     # ensure local GPU is idle before timing
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            fn()
            t1.record()
            torch.cuda.synchronize()
            times.append(t0.elapsed_time(t1))
        return times

    all_results = {}

    for M in ALL_M:
        M_local = M // world_size
        if M_local < CHUNK or M_local % CHUNK != 0:
            continue

        ws_obj = GemmReduceScatterWorkspace(M=M, N=N, K_local=K_local,
                                            group=group, dtype=DTYPE, device=device)

        X = symm_mem.empty(M, K_local, device=device, dtype=DTYPE).normal_()
        W = torch.randn(K_local, N, device=device, dtype=DTYPE)
        out_naive = torch.empty(M_local, N, device=device, dtype=DTYPE)

        fns = {}
        if "naive" in BACKENDS:
            def _naive():
                partial = X @ W
                dist.reduce_scatter_tensor(out_naive, partial, group=group)
            fns["naive"] = _naive

        if "fused" in BACKENDS:
            # PyTorch built-in fused_matmul_reduce_scatter — the vLLM production
            # baseline.  Uses "sum" reduction to match our kernel's output.
            # X must be a symm_mem tensor (already allocated above).
            fns["fused"] = lambda: torch.ops.symm_mem.fused_matmul_reduce_scatter(
                X, W, "sum", scatter_dim=0, group_name=group.group_name
            )

        if "spin" in BACKENDS:
            fns["spin"] = lambda: gemm_reduce_scatter(X, W, group, backend="triton",
                                                       workspace=ws_obj)

        if "mbarrier" in BACKENDS:
            def _mbarrier():
                saved = Configs.BARRIER_MODE
                Configs.BARRIER_MODE = "mbarrier"
                try:
                    return gemm_reduce_scatter(X, W, group, backend="triton", workspace=ws_obj)
                finally:
                    Configs.BARRIER_MODE = saved
            fns["mbarrier"] = _mbarrier

        # Warmup (all backends, barrier+sync after each backend)
        for b in BACKENDS:
            if b not in fns:
                continue
            for _ in range(WARMUP):
                fns[b]()
            dist.barrier(group)
            torch.cuda.synchronize()

        # Measurement — barrier before each backend to ensure clean start
        stats_per_backend = {}
        for b in BACKENDS:
            if b not in fns:
                continue
            dist.barrier(group)
            torch.cuda.synchronize()
            raw = _time_calls(fns[b], N_ITERS)
            dist.barrier(group)
            stats_per_backend[b] = _compute_stats(raw)

        ws_obj.destroy()

        if rank == 0:
            row = f"  {M:>7}"
            for b in BACKENDS:
                if b not in stats_per_backend:
                    row += f"  {'N/A':>{col_w}}"
                    continue
                s = stats_per_backend[b]
                row += f"  {s['mean']:6.3f}±{s['std']:5.3f}"
            # median column (first non-naive backend or naive)
            ref_b = "spin" if "spin" in stats_per_backend else list(stats_per_backend.keys())[0]
            row += f"  {stats_per_backend[ref_b]['median']:9.3f}"
            # speedup over naive
            if "naive" in stats_per_backend:
                naive_mean = stats_per_backend["naive"]["mean"]
                for b in [x for x in BACKENDS if x != "naive"]:
                    if b in stats_per_backend:
                        row += f"  {naive_mean/stats_per_backend[b]['mean']:>10.2f}x"
            log(row)
            all_results[M] = stats_per_backend

    # ------------------------------------------------------------------
    # Per-backend min/max/p5/p95 detail table
    # ------------------------------------------------------------------
    if rank == 0 and all_results:
        log("")
        log("  Detailed statistics (ms):")
        log(f"  {'M':>7}  {'backend':>9}  {'mean':>7}  {'std':>6}  {'min':>7}  {'p5':>7}  "
            f"{'median':>7}  {'p95':>7}  {'max':>7}")
        log(f"  {'-'*80}")
        for M, sb in all_results.items():
            for b, s in sb.items():
                log(f"  {M:>7}  {b:>9}  {s['mean']:7.3f}  {s['std']:6.3f}  "
                    f"{s['min']:7.3f}  {s['p5']:7.3f}  {s['median']:7.3f}  "
                    f"{s['p95']:7.3f}  {s['max']:7.3f}")
        log("")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    if rank == 0 and all_results:
        result = {
            "timestamp": datetime.now().isoformat(),
            "world_size": world_size,
            "K": K, "N": N, "dtype": "bfloat16",
            "n_iters": N_ITERS, "warmup": WARMUP,
            "backends": BACKENDS,
            "results": {
                str(M): {
                    b: {k: v for k, v in s.items() if k != "raw"}
                    for b, s in sb.items()
                }
                for M, sb in all_results.items()
            },
            "raw_times": {
                str(M): {b: s["raw"] for b, s in sb.items()}
                for M, sb in all_results.items()
            },
        }
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        log(f"  Results saved to: {result_path}")
        if _log:
            _log.close()

    dist.destroy_process_group()


def run_ws(world_size, gpus):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = str(RESULTS_DIR / f"ws{world_size}_{ts}.json")
    log_path    = str(RESULTS_DIR / f"ws{world_size}_{ts}.txt")
    with socket.socket() as s:
        s.bind(("", 0)); port = s.getsockname()[1]
    multiprocessing.set_start_method("spawn", force=True)
    procs = [multiprocessing.Process(target=worker,
                                     args=(world_size, r, port, result_path, log_path))
             for r in range(world_size)]
    for p in procs: p.start()
    for p in procs: p.join(timeout=WORKER_TIMEOUT_S)
    alive = [p for p in procs if p.is_alive()]
    if alive:
        for p in alive: p.kill()
        print(f"  ws={world_size}: workers timed out after {WORKER_TIMEOUT_S}s — killed", flush=True)
    elif any(p.exitcode != 0 for p in procs):
        print(f"  ws={world_size}: a worker failed (exit codes: {[p.exitcode for p in procs]})",
              flush=True)
    else:
        print(f"  ws={world_size}: done → {result_path}", flush=True)


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1: Pre-warming Triton kernel cache on GPU 0")
    print("=" * 60, flush=True)
    prewarm()

    print("=" * 60)
    print(f"Phase 2: Benchmark — backends={BACKENDS}")
    print(f"         n_iters={N_ITERS}  warmup={WARMUP}  M={ALL_M}")
    print(f"Results: {RESULTS_DIR}")
    print("=" * 60, flush=True)

    run_ws(2, [0, 1])
    run_ws(4, [0, 1, 2, 3])
    # ws=8 requires GPU 4 to be healthy; skip until hardware is reset.
    # run_ws(8, list(range(8)))

    print("\nDone.", flush=True)
