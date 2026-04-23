# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for GEMM + Reduce-Scatter (flashinfer.comm.gemm_reduce_scatter).

Run with pytest:
    pytest tests/comm/test_gemm_reduce_scatter.py -v

Or directly:
    python tests/comm/test_gemm_reduce_scatter.py --world_size 4 --correctness
"""

import argparse
import multiprocessing as mp
import os
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from flashinfer.comm.gemm_reduce_scatter import gemm_reduce_scatter


# ---------------------------------------------------------------------------
# Utilities — matching pattern from test_vllm_custom_allreduce.py
# ---------------------------------------------------------------------------

def get_open_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]
    except OSError:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("::1", 0))
            return s.getsockname()[1]


def multi_process_parallel(world_size: int, target, target_args: tuple = ()) -> None:
    """Launch world_size processes, each calling target(world_size, rank, port, *target_args)."""
    mp.set_start_method("spawn", force=True)
    # Ensure the repo root is on PYTHONPATH so spawn'd processes can import
    # the tests package (tests/__init__.py exists but repo root may not be
    # on sys.path when pytest uses --import-mode=importlib).
    repo_root = str(Path(__file__).resolve().parents[2])
    existing = os.environ.get("PYTHONPATH", "")
    if repo_root not in (existing.split(":") if existing else []):
        os.environ["PYTHONPATH"] = f"{repo_root}:{existing}" if existing else repo_root
    port = get_open_port()
    procs = []
    for rank in range(world_size):
        proc = mp.Process(
            target=target,
            args=(world_size, rank, port) + target_args,
            name=f"Worker-{rank}",
        )
        proc.start()
        procs.append(proc)
    for rank, proc in enumerate(procs):
        proc.join()
        assert proc.exitcode == 0, f"Process {rank} failed with exit code {proc.exitcode}"


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------

def _ref_gemm_reduce_scatter(X_local, W_local, group):
    """
    fp32 ground truth for correctness comparison.

    Uses fp32 matmul (not bf16) and fp32 all_gather+sum to avoid NCCL's
    bf16 ring-reduce rounding (which accumulates bf16 rounding error and
    would make a strict comparison unfair to our fp32-accumulating kernel).
    """
    world_size = dist.get_world_size(group)
    rank       = dist.get_rank(group)
    M_local    = X_local.shape[0] // world_size

    # fp32 partial K-sum, then all_gather + sum on each rank
    partial_fp32 = X_local.float() @ W_local.float()
    all_partials = [torch.zeros_like(partial_fp32) for _ in range(world_size)]
    dist.all_gather(all_partials, partial_fp32)
    gt_fp32 = sum(all_partials)[rank * M_local : (rank + 1) * M_local]
    return gt_fp32.to(X_local.dtype)


# ---------------------------------------------------------------------------
# Distributed initialisation
# ---------------------------------------------------------------------------

def _dist_init(world_size: int, rank: int, port: int):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD
    symm_mem.enable_symm_mem_for_group(group.group_name)
    return device, group


# ---------------------------------------------------------------------------
# Module-level worker functions (must be at top-level for mp.Process)
# ---------------------------------------------------------------------------

def _correctness_worker(world_size, rank, port, dtype_str, M, K, N, atol, rtol, backend="triton"):
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
    device, group = _dist_init(world_size, rank, port)
    try:
        torch.manual_seed(rank)
        K_local = K // world_size
        X_local = torch.randn(M, K_local, device=device, dtype=dtype)
        W_local = torch.randn(K_local, N, device=device, dtype=dtype)

        out = gemm_reduce_scatter(X_local, W_local, group, backend=backend, verbose=(rank == 0))
        ref = _ref_gemm_reduce_scatter(X_local, W_local, group)
        torch.cuda.synchronize()

        if not torch.allclose(out.float(), ref.float(), atol=atol, rtol=rtol):
            max_err = (out.float() - ref.float()).abs().max().item()
            raise AssertionError(
                f"rank={rank}: max_abs_err={max_err:.4e} (atol={atol}, rtol={rtol})"
            )
        if rank == 0:
            print(f"[correctness] dtype={dtype} M={M} K={K} N={N} ws={world_size} backend={backend} PASSED")
    finally:
        dist.destroy_process_group()


def _benchmark_worker(world_size, rank, port, M, K, N, n_iters=50, warmup=10, backend="triton"):
    device, group = _dist_init(world_size, rank, port)
    try:
        K_local = K // world_size
        torch.manual_seed(0)
        X_local = torch.randn(M, K_local, device=device, dtype=torch.bfloat16)
        W_local = torch.randn(K_local, N, device=device, dtype=torch.bfloat16)

        for _ in range(warmup):
            _ref_gemm_reduce_scatter(X_local, W_local, group)
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(n_iters):
            _ref_gemm_reduce_scatter(X_local, W_local, group)
        t1.record()
        torch.cuda.synchronize()
        ref_ms = t0.elapsed_time(t1) / n_iters

        for _ in range(warmup):
            gemm_reduce_scatter(X_local, W_local, group, backend=backend)
        torch.cuda.synchronize()
        t0.record()
        for _ in range(n_iters):
            gemm_reduce_scatter(X_local, W_local, group, backend=backend)
        t1.record()
        torch.cuda.synchronize()
        ours_ms = t0.elapsed_time(t1) / n_iters

        if rank == 0:
            print(
                f"[bench] M={M} K={K} N={N} ws={world_size} backend={backend} | "
                f"ours={ours_ms:.3f}ms ref={ref_ms:.3f}ms "
                f"speedup={ref_ms / ours_ms:.2f}x"
            )
    finally:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Skip helpers — called in the main pytest process only, never inside workers
# ---------------------------------------------------------------------------

def _skip_if_insufficient_gpus(world_size: int):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Need {world_size} GPUs, found {torch.cuda.device_count()}")


def _gpu_major_via_smi() -> int:
    """Return compute capability major of GPU 0 via nvidia-smi.

    Does NOT initialize the CUDA runtime, so it is safe to call before mp.spawn
    (spawn requires that CUDA is not yet initialized in the parent process).
    """
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader", "--id=0"],
            timeout=5,
        ).decode().strip()
        return int(out.split(".")[0])
    except Exception:
        return 0


def _skip_if_sm_too_high():
    """Skip if GPU 0 is SM >= 100 (cuTile backend not yet implemented).

    Uses nvidia-smi instead of torch.cuda.get_device_capability to avoid
    initializing the CUDA runtime before mp.spawn.
    """
    if _gpu_major_via_smi() >= 10:
        pytest.skip("Triton backend targets SM < 100; cuTile backend not yet implemented")


# ---------------------------------------------------------------------------
# Pytest tests
# ---------------------------------------------------------------------------

CORRECTNESS_CASES = [
    # (M,    K,    N,    dtype_str,   atol,  rtol)
    # atol=3.0: accommodates bfloat16 quantisation of GEMM outputs.
    # The fp32 reference differs from bf16 results by up to ~1 bf16 ULP
    # per output element (~0.35 for values near 45, ~2.0 for values near 256).
    # With world_size partials accumulated, max error can reach ~3.0.
    (4096,  8192, 2048, "bfloat16",  3.0,   0.05),
    (8192,  8192, 2048, "bfloat16",  3.0,   0.05),
    (16384, 8192, 2048, "bfloat16",  3.0,   0.05),
    (4096,  4096, 4096, "float16",   3.0,   0.05),
]


@pytest.mark.parametrize("backend", ["triton", "cublas"])
@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("M,K,N,dtype_str,atol,rtol", CORRECTNESS_CASES)
def test_correctness(world_size, M, K, N, dtype_str, atol, rtol, backend):
    _skip_if_insufficient_gpus(world_size)
    _skip_if_sm_too_high()
    if K % world_size != 0:
        pytest.skip(f"K={K} not divisible by world_size={world_size}")
    multi_process_parallel(
        world_size,
        _correctness_worker,
        target_args=(dtype_str, M, K, N, atol, rtol, backend),
    )


@pytest.mark.parametrize("backend", ["triton", "cublas"])
@pytest.mark.parametrize("world_size", [2, 4])
def test_benchmark(world_size, backend):
    _skip_if_insufficient_gpus(world_size)
    _skip_if_sm_too_high()
    for M in [4096, 8192, 16384, 32768]:
        multi_process_parallel(
            world_size,
            _benchmark_worker,
            target_args=(M, 8192, 2048, 50, 10, backend),
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli_correctness_worker(world_size, rank, port, dtype_str):
    dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float16
    device, group = _dist_init(world_size, rank, port)
    try:
        for M, K, N in [(4096, 8192, 2048), (8192, 8192, 2048), (16384, 8192, 2048)]:
            if K % world_size != 0:
                continue
            K_local = K // world_size
            torch.manual_seed(rank)
            X_local = torch.randn(M, K_local, device=device, dtype=dtype)
            W_local = torch.randn(K_local, N, device=device, dtype=dtype)
            out = gemm_reduce_scatter(X_local, W_local, group, verbose=(rank == 0))
            ref = _ref_gemm_reduce_scatter(X_local, W_local, group)
            torch.cuda.synchronize()
            if not torch.allclose(out.float(), ref.float(), atol=0.05, rtol=0.05):
                max_err = (out.float() - ref.float()).abs().max().item()
                raise AssertionError(f"rank={rank} M={M} K={K} N={N}: err={max_err:.4e}")
            if rank == 0:
                print(f"[correctness] M={M} K={K} N={N} ws={world_size} PASSED")
    finally:
        dist.destroy_process_group()


def _cli_benchmark_worker(world_size, rank, port, dtype_str):
    dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float16
    device, group = _dist_init(world_size, rank, port)
    try:
        for M in [1024, 2048, 4096, 8192, 16384, 32768, 65536]:
            K, N = 8192, 2048
            if K % world_size != 0:
                continue
            K_local = K // world_size
            torch.manual_seed(0)
            X_local = torch.randn(M, K_local, device=device, dtype=dtype)
            W_local = torch.randn(K_local, N, device=device, dtype=dtype)
            for _ in range(5):
                gemm_reduce_scatter(X_local, W_local, group)
            torch.cuda.synchronize()
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            for _ in range(20):
                gemm_reduce_scatter(X_local, W_local, group)
            t1.record()
            torch.cuda.synchronize()
            if rank == 0:
                print(f"[bench] M={M} K={K} N={N} ws={world_size} {t0.elapsed_time(t1)/20:.3f}ms")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count())
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    args = parser.parse_args()

    if not args.correctness and not args.benchmark:
        args.correctness = True
        args.benchmark = True

    if args.correctness:
        multi_process_parallel(
            args.world_size, _cli_correctness_worker, target_args=(args.dtype,)
        )
    if args.benchmark:
        multi_process_parallel(
            args.world_size, _cli_benchmark_worker, target_args=(args.dtype,)
        )
