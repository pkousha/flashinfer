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
Triton GEMM + Reduce-Scatter: overlapped compute and communication.

See design.md for the full algorithm design, cache footprint analysis,
and mbarrier rationale.  See comparison.md for benchmark results.
"""

import warnings

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

from .configs import Configs

# ---------------------------------------------------------------------------
# Default tile configuration (tuned for bf16/fp16, Hopper / Ampere)
# ---------------------------------------------------------------------------
_BLOCK_M      = 128
_BLOCK_N      = 128
_BLOCK_K      = 64
_GROUP_SIZE_M = 8

# CHUNK_SIZE_M: M rows per (chunk, dest) tile-set.
# Smaller → finer overlap and more signals, but more inter-CTA barrier overhead.
# Must be a multiple of BLOCK_M.
_CHUNK_SIZE_M = _GROUP_SIZE_M * _BLOCK_M   # 1024

# Cache-line padding constants (see design.md §7 and configs.py).
_ARRIVE_STRIDE = 32   # padding per arrive_count slot (one 128-B cache line)
_SIGNAL_STRIDE = 32   # padding per rs_signal slot    (one 128-B cache line)


# ---------------------------------------------------------------------------
# Utility: 2-D swizzled tile index
# ---------------------------------------------------------------------------

@triton.jit
def _swizzle_2d_from_bid(num_tiles_m, num_tiles_n, GROUP_SIZE_M: tl.constexpr, bid):
    """Map a linear block-id to (tile_m, tile_n) with L2-friendly swizzle."""
    num_in_group = GROUP_SIZE_M * num_tiles_n
    group_id     = bid // num_in_group
    first_m      = group_id * GROUP_SIZE_M
    group_sz_m   = tl.minimum(num_tiles_m - first_m, GROUP_SIZE_M)
    tile_m       = first_m + (bid % group_sz_m)
    tile_n       = (bid % num_in_group) // group_sz_m
    return tile_m, tile_n


# ---------------------------------------------------------------------------
# Barrier helpers
# ---------------------------------------------------------------------------

@triton.jit
def _spin_load(ptr):
    """Volatile global load for spin-wait polling (all SM generations)."""
    return tl.load(ptr, volatile=True)


@triton.jit
def _acquire_load(ptr):
    """
    System-scope acquire load for spin-wait polling on SM >= 90 (Hopper).

    Uses PTX ld.global.acquire.sys.b32 instead of ld.volatile.global.b32.
    Provides hardware-level system-scope acquire semantics, pairing correctly
    with the sem="release" atomic_add in the arrive barrier.

    Both orderings (ld.global.acquire.sys and ld.acquire.sys.global) are
    accepted by ptxas — confirmed from multiple production files in FlashInfer.
    """
    return tl.inline_asm_elementwise(
        "ld.global.acquire.sys.b32 $0, [$1];",
        "=r, l",
        args=[ptr],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


# ---------------------------------------------------------------------------
# Kernel 1: Barrier
# ---------------------------------------------------------------------------

@triton.jit
def _barrier_triton_kernel(
    signal_pads,                   # tuple[Tensor] — one (world_size,) pad per rank
    rank: tl.constexpr,
    world_size: tl.constexpr,
):
    """
    Deposit-and-withdraw barrier.  Grid = (1,) — single block.

    Identical to the barrier in the AG+GEMM reference (PR #2665).

    Phase 1 — Deposit (loop over all peers):
      For each peer p: atomically set peer_p.barrier_pad[rank] 0 → 1.
      tl.static_range ensures peer is a compile-time constant so Triton
      can index into the signal_pads tuple.

    Phase 2 — Withdraw (loop over all peers):
      For each peer p: spin-wait on own_pad[p] until peer p deposits 1,
      then clear it to 0 for future reuse.
    """
    for peer in tl.static_range(world_size):
        deposit_pad = signal_pads[peer]
        while tl.atomic_cas(deposit_pad + rank, 0, 1, sem="release") != 0:
            pass

    withdraw_pad = signal_pads[rank]
    for peer in tl.static_range(world_size):
        while tl.atomic_cas(withdraw_pad + peer, 1, 0, sem="acquire") != 1:
            pass


def _barrier_triton(barrier_sigpads: tuple, rank: int) -> None:
    world_size = len(barrier_sigpads)
    _barrier_triton_kernel[(1,)](
        barrier_sigpads,
        rank=rank,
        world_size=world_size,
    )


# ---------------------------------------------------------------------------
# Kernel 2: GEMM+Push  (single persistent kernel, Design B)
# ---------------------------------------------------------------------------

@triton.jit
def _gemm_push_kernel(
    x_ptr, w_ptr,
    remote_recv_ptrs,    # tuple[WORLD_SIZE] — (M_local, N) recv slots
    remote_signal_ptrs,  # tuple[WORLD_SIZE] — (num_chunks, SIGNAL_STRIDE) signal pads
    arrive_count_ptr,    # (num_chunks * WORLD_SIZE * ARRIVE_STRIDE,) int32 counters
    M_local, N, K_local, num_chunks,
    CHUNK_SIZE_M: tl.constexpr,
    ARRIVE_STRIDE: tl.constexpr,
    SIGNAL_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    """
    Design B GEMM+Push kernel — chunk-outer, dest-inner, spin barrier.

    For each (chunk, dest_shift) pair, all CTAs compute assigned GEMM tiles
    and store directly to dest's recv_buf[RANK] via NVLink peer memory.
    After all CTAs finish the tile set, the last CTA fires the signal.
    Inter-CTA arrive barrier uses padded unique counter slots.
    """
    pid          = tl.program_id(0)
    num_programs = tl.num_programs(0)

    tiles_per_chunk_m = CHUNK_SIZE_M // BLOCK_M
    num_tiles_n       = tl.cdiv(N, BLOCK_N)
    tiles_per_chunk   = tiles_per_chunk_m * num_tiles_n

    for chunk in range(num_chunks):
        chunk_m_off = chunk * CHUNK_SIZE_M

        for dest_shift in tl.static_range(WORLD_SIZE):
            dest          = (RANK + dest_shift) % WORLD_SIZE
            remote_recv   = remote_recv_ptrs[dest_shift]
            remote_signal = remote_signal_ptrs[dest_shift]
            x_m_base      = dest * M_local + chunk_m_off

            tile_id = pid
            while tile_id < tiles_per_chunk:
                tile_m_in_chunk, tile_n = _swizzle_2d_from_bid(
                    tiles_per_chunk_m, num_tiles_n, GROUP_SIZE_M, tile_id
                )
                offs_am = x_m_base + tile_m_in_chunk * BLOCK_M + tl.arange(0, BLOCK_M)
                offs_bn = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)

                acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                for k in range(tl.cdiv(K_local, BLOCK_K)):
                    k_off  = k * BLOCK_K
                    offs_k = tl.arange(0, BLOCK_K)
                    x_mask = (offs_am[:, None] < x_m_base + CHUNK_SIZE_M) & \
                             ((k_off + offs_k)[None, :] < K_local)
                    w_mask = ((k_off + offs_k)[:, None] < K_local) & \
                             (offs_bn[None, :] < N)
                    x = tl.load(
                        x_ptr + offs_am[:, None] * K_local + (k_off + offs_k)[None, :],
                        mask=x_mask, other=0.0,
                    )
                    w = tl.load(
                        w_ptr + (k_off + offs_k)[:, None] * N + offs_bn[None, :],
                        mask=w_mask, other=0.0,
                    )
                    acc = tl.dot(x, w, acc)

                out_m    = chunk_m_off + tile_m_in_chunk * BLOCK_M + tl.arange(0, BLOCK_M)
                out_n    = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
                out_mask = (out_m[:, None] < M_local) & (out_n[None, :] < N)
                tl.store(
                    remote_recv + out_m[:, None] * N + out_n[None, :],
                    acc.to(remote_recv.dtype.element_ty),
                    mask=out_mask,
                )
                tile_id += num_programs

            ctr_idx    = (chunk * WORLD_SIZE + dest_shift) * ARRIVE_STRIDE
            prev_count = tl.atomic_add(arrive_count_ptr + ctr_idx, 1, sem="release")

            if prev_count == num_programs - 1:
                tl.atomic_xchg(
                    remote_signal + chunk * SIGNAL_STRIDE,
                    1,
                    sem="release",
                )

            while tl.load(arrive_count_ptr + ctr_idx, volatile=True) < num_programs:
                pass


@triton.jit
def _gemm_push_kernel_mbarrier(
    x_ptr, w_ptr,
    remote_recv_ptrs,
    remote_signal_ptrs,
    arrive_count_ptr,
    M_local, N, K_local, num_chunks,
    CHUNK_SIZE_M: tl.constexpr,
    ARRIVE_STRIDE: tl.constexpr,
    SIGNAL_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    """
    Design B GEMM+Push — mbarrier variant for SM >= 90 (Hopper).

    Identical to _gemm_push_kernel except the arrive-barrier spin loop uses
    ld.global.acquire.sys instead of ld.volatile.global.  On Hopper, acquire.sys
    semantics allow the hardware coherence protocol to serve repeated reads more
    efficiently than volatile.

    See design.md §8 for detailed mbarrier discussion and limitations.
    """
    pid          = tl.program_id(0)
    num_programs = tl.num_programs(0)

    tiles_per_chunk_m = CHUNK_SIZE_M // BLOCK_M
    num_tiles_n       = tl.cdiv(N, BLOCK_N)
    tiles_per_chunk   = tiles_per_chunk_m * num_tiles_n

    for chunk in range(num_chunks):
        chunk_m_off = chunk * CHUNK_SIZE_M

        for dest_shift in tl.static_range(WORLD_SIZE):
            dest          = (RANK + dest_shift) % WORLD_SIZE
            remote_recv   = remote_recv_ptrs[dest_shift]
            remote_signal = remote_signal_ptrs[dest_shift]
            x_m_base      = dest * M_local + chunk_m_off

            tile_id = pid
            while tile_id < tiles_per_chunk:
                tile_m_in_chunk, tile_n = _swizzle_2d_from_bid(
                    tiles_per_chunk_m, num_tiles_n, GROUP_SIZE_M, tile_id
                )
                offs_am = x_m_base + tile_m_in_chunk * BLOCK_M + tl.arange(0, BLOCK_M)
                offs_bn = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)

                acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                for k in range(tl.cdiv(K_local, BLOCK_K)):
                    k_off  = k * BLOCK_K
                    offs_k = tl.arange(0, BLOCK_K)
                    x_mask = (offs_am[:, None] < x_m_base + CHUNK_SIZE_M) & \
                             ((k_off + offs_k)[None, :] < K_local)
                    w_mask = ((k_off + offs_k)[:, None] < K_local) & \
                             (offs_bn[None, :] < N)
                    x = tl.load(
                        x_ptr + offs_am[:, None] * K_local + (k_off + offs_k)[None, :],
                        mask=x_mask, other=0.0,
                    )
                    w = tl.load(
                        w_ptr + (k_off + offs_k)[:, None] * N + offs_bn[None, :],
                        mask=w_mask, other=0.0,
                    )
                    acc = tl.dot(x, w, acc)

                out_m    = chunk_m_off + tile_m_in_chunk * BLOCK_M + tl.arange(0, BLOCK_M)
                out_n    = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
                out_mask = (out_m[:, None] < M_local) & (out_n[None, :] < N)
                tl.store(
                    remote_recv + out_m[:, None] * N + out_n[None, :],
                    acc.to(remote_recv.dtype.element_ty),
                    mask=out_mask,
                )
                tile_id += num_programs

            ctr_idx    = (chunk * WORLD_SIZE + dest_shift) * ARRIVE_STRIDE
            prev_count = tl.atomic_add(arrive_count_ptr + ctr_idx, 1, sem="release")

            if prev_count == num_programs - 1:
                tl.atomic_xchg(
                    remote_signal + chunk * SIGNAL_STRIDE,
                    1,
                    sem="release",
                )

            while _acquire_load(arrive_count_ptr + ctr_idx) < tl.cast(
                num_programs, tl.uint32
            ):
                pass


# ---------------------------------------------------------------------------
# Kernel 3: Wait+Reduce
# ---------------------------------------------------------------------------

@triton.jit
def _wait_reduce_kernel(
    recv_buf_ptr,       # MY recv_buf  (world_size, M_local, N)
    rs_signal_ptr,      # MY rs_signal_buf  (world_size, num_chunks, SIGNAL_STRIDE)
    out_ptr,            # output  (M_local, N)
    M_local, N, num_chunks,
    stride_per_src,     # M_local * N
    CHUNK_SIZE_M: tl.constexpr,
    SIGNAL_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    """
    Persistent tile-based Wait+Reduce kernel.

    For every output tile (tile_m, tile_n):
      1. Determine which chunk the tile belongs to.
      2. For each source rank (starting from self):
           Spin-wait on rs_signal[src, chunk] (SIGNAL_STRIDE-padded, one cache line).
           Load recv_buf[src][tile] and accumulate into fp32.
      3. Store accumulated result to output.
    """
    pid          = tl.program_id(0)
    num_programs = tl.num_programs(0)

    tiles_per_chunk_m = CHUNK_SIZE_M // BLOCK_M
    num_tiles_m       = tl.cdiv(M_local, BLOCK_M)
    num_tiles_n       = tl.cdiv(N, BLOCK_N)
    total_tiles       = num_tiles_m * num_tiles_n

    tile_id = pid
    while tile_id < total_tiles:
        tile_m = tile_id // num_tiles_n
        tile_n = tile_id % num_tiles_n

        chunk  = tile_m // tiles_per_chunk_m

        offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask   = (offs_m[:, None] < M_local) & (offs_n[None, :] < N)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for shift in tl.static_range(WORLD_SIZE):
            src = (RANK + shift) % WORLD_SIZE

            sig_off = (src * num_chunks + chunk) * SIGNAL_STRIDE
            while tl.load(rs_signal_ptr + sig_off, volatile=True) == 0:
                pass

            partial = tl.load(
                recv_buf_ptr
                + src * stride_per_src
                + offs_m[:, None] * N
                + offs_n[None, :],
                mask=mask, other=0.0,
            )
            acc += partial.to(tl.float32)

        tl.store(
            out_ptr + offs_m[:, None] * N + offs_n[None, :],
            acc.to(out_ptr.dtype.element_ty),
            mask=mask,
        )
        tile_id += num_programs


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def gemm_reduce_scatter_triton(
    X_local: torch.Tensor,
    W_local: torch.Tensor,
    group: dist.ProcessGroup,
    *,
    verbose: bool = False,
    workspace=None,   # Optional[GemmReduceScatterWorkspace]
) -> torch.Tensor:
    """
    Triton GEMM + Reduce-Scatter for SM < 100 (Hopper, Ampere).

    Uses a single persistent Triton kernel (GEMM+Push) that computes GEMM
    tiles and writes them directly to remote recv_buf via NVLink peer memory,
    with an inter-CTA arrive barrier per (chunk, dest) pair.

    Parameters
    ----------
    X_local   : (M, K_local) — K-sharded activation, contiguous, bf16/fp16.
    W_local   : (K_local, N) — K-sharded weight, contiguous, same dtype.
    group     : process group for the reduce-scatter.
    workspace : GemmReduceScatterWorkspace (optional).  If provided, uses
                pre-allocated buffers/streams — eliminates 3× rendezvous
                per call.  If None, allocates everything per-call (original
                behaviour, backward-compatible).

    Returns
    -------
    (M_local, N) — this rank's slice of sum_r(X_local_r @ W_local_r).
    """
    Configs.initialize()

    assert X_local.is_contiguous(), "X_local must be contiguous"
    assert W_local.is_contiguous(), "W_local must be contiguous"

    M, K_local = X_local.shape
    K_w,     N = W_local.shape
    assert K_w == K_local, f"K mismatch: X has {K_local}, W has {K_w}"

    world_size = dist.get_world_size(group)
    rank       = dist.get_rank(group)

    assert M % world_size == 0, f"M ({M}) must be divisible by world_size ({world_size})"
    M_local = M // world_size

    CHUNK_SIZE_M  = _CHUNK_SIZE_M
    ARRIVE_STRIDE = _ARRIVE_STRIDE
    SIGNAL_STRIDE = _SIGNAL_STRIDE

    assert M_local % CHUNK_SIZE_M == 0, (
        f"M_local ({M_local}) must be divisible by CHUNK_SIZE_M ({CHUNK_SIZE_M})"
    )
    assert CHUNK_SIZE_M % _BLOCK_M == 0
    num_chunks = M_local // CHUNK_SIZE_M

    device = X_local.device
    dtype  = X_local.dtype

    # ------------------------------------------------------------------
    # Symmetric memory — use workspace if provided, else allocate per-call
    # ------------------------------------------------------------------
    if workspace is not None:
        assert workspace.is_compatible(M, N, K_local, world_size, dtype, device), (
            f"Workspace dimensions mismatch. "
            f"Workspace: M={workspace.M} N={workspace.N} K_local={workspace.K_local} "
            f"ws={workspace.world_size} dtype={workspace.dtype} device={workspace.device}. "
            f"Call: M={M} N={N} K_local={K_local} ws={world_size} dtype={dtype} device={device}."
        )
        barrier_sigpads    = workspace.barrier_sigpads
        recv_buf           = workspace.recv_buf
        rs_signal_buf      = workspace.rs_signal_buf
        remote_recv_ptrs   = workspace.remote_recv_ptrs
        remote_signal_ptrs = workspace.remote_signal_ptrs
        arrive_count       = workspace.arrive_count
    else:
        # Original per-call allocation (backward-compatible)
        barrier_buf    = symm_mem.empty(world_size, device=device, dtype=Configs.SIGNAL_DTYPE)
        barrier_handle = symm_mem.rendezvous(barrier_buf, group.group_name)
        barrier_sigpads = tuple(
            barrier_handle.get_signal_pad(r, (world_size,), Configs.SIGNAL_DTYPE, 0)
            for r in range(world_size)
        )

        recv_buf = symm_mem.empty(world_size, M_local, N, device=device, dtype=dtype)
        handle   = symm_mem.rendezvous(recv_buf, group.group_name)

        rs_signal_buf    = symm_mem.empty(
            world_size, num_chunks, SIGNAL_STRIDE,
            device=device, dtype=Configs.SIGNAL_DTYPE,
        )
        rs_signal_handle = symm_mem.rendezvous(rs_signal_buf, group.group_name)

        remote_recv_ptrs = tuple(
            handle.get_buffer(
                (rank + shift) % world_size,
                (world_size, M_local, N),
                dtype,
            )[rank]
            for shift in range(world_size)
        )

        remote_signal_ptrs = tuple(
            rs_signal_handle.get_buffer(
                (rank + shift) % world_size,
                (world_size, num_chunks, SIGNAL_STRIDE),
                Configs.SIGNAL_DTYPE,
            )[rank]
            for shift in range(world_size)
        )

        arrive_count = torch.zeros(
            num_chunks * world_size * ARRIVE_STRIDE,
            dtype=torch.int32,
            device=device,
        )

    # ------------------------------------------------------------------
    # Tile and SM configuration
    # ------------------------------------------------------------------
    BLOCK_M      = _BLOCK_M
    BLOCK_N      = _BLOCK_N
    BLOCK_K      = _BLOCK_K
    GROUP_SIZE_M = _GROUP_SIZE_M

    num_tiles_m_out = triton.cdiv(M_local, BLOCK_M)
    num_tiles_n_out = triton.cdiv(N, BLOCK_N)
    total_out_tiles = num_tiles_m_out * num_tiles_n_out

    NUM_SMS = torch.cuda.get_device_properties(device).multi_processor_count

    reserve_reduce  = max(1, NUM_SMS // 4)
    NUM_SMS_compute = NUM_SMS - reserve_reduce

    gemm_grid   = (NUM_SMS_compute,)
    reduce_grid = (min(reserve_reduce, total_out_tiles),)

    # ------------------------------------------------------------------
    # Barrier mode (FLASHINFER_GRS_BARRIER env var)
    # ------------------------------------------------------------------
    use_mbarrier = False
    barrier_mode = Configs.BARRIER_MODE if hasattr(Configs, "BARRIER_MODE") else "spin"
    if barrier_mode == "mbarrier":
        major, _ = torch.cuda.get_device_capability(device)
        if major < 9:
            warnings.warn(
                "FLASHINFER_GRS_BARRIER=mbarrier requires SM >= 90. "
                "Falling back to spin.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            use_mbarrier = True

    main_stream = torch.cuda.current_stream()

    if verbose and rank == 0:
        print(
            f"[gemm_reduce_scatter_triton] M={M} K_local={K_local} N={N} "
            f"ws={world_size} M_local={M_local} num_chunks={num_chunks} "
            f"CHUNK={CHUNK_SIZE_M} BLOCK=({BLOCK_M},{BLOCK_N},{BLOCK_K}) "
            f"SMS: compute={NUM_SMS_compute} reduce={reserve_reduce} "
            f"barrier={'mbarrier' if use_mbarrier else 'spin'}"
        )

    # ------------------------------------------------------------------
    # Kernel 1: Barrier
    # ------------------------------------------------------------------
    _barrier_triton(barrier_sigpads, rank)

    # ------------------------------------------------------------------
    # Kernel 2: GEMM+Push
    # ------------------------------------------------------------------
    compute_stream = workspace.compute_stream if workspace is not None else torch.cuda.Stream()
    compute_stream.wait_stream(main_stream)

    kernel_kwargs = dict(
        M_local=M_local, N=N, K_local=K_local, num_chunks=num_chunks,
        CHUNK_SIZE_M=CHUNK_SIZE_M,
        ARRIVE_STRIDE=ARRIVE_STRIDE,
        SIGNAL_STRIDE=SIGNAL_STRIDE,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        RANK=rank, WORLD_SIZE=world_size,
    )

    with torch.cuda.stream(compute_stream):
        if use_mbarrier:
            _gemm_push_kernel_mbarrier[gemm_grid](
                X_local, W_local,
                remote_recv_ptrs,
                remote_signal_ptrs,
                arrive_count,
                **kernel_kwargs,
            )
        else:
            _gemm_push_kernel[gemm_grid](
                X_local, W_local,
                remote_recv_ptrs,
                remote_signal_ptrs,
                arrive_count,
                **kernel_kwargs,
            )

    # ------------------------------------------------------------------
    # Kernel 3: Wait+Reduce (concurrent with compute_stream)
    # ------------------------------------------------------------------
    reduce_stream = workspace.reduce_stream if workspace is not None else torch.cuda.Stream()
    reduce_stream.wait_stream(main_stream)

    output = torch.empty(M_local, N, device=device, dtype=dtype)

    with torch.cuda.stream(reduce_stream):
        _wait_reduce_kernel[reduce_grid](
            recv_buf, rs_signal_buf, output,
            M_local=M_local, N=N, num_chunks=num_chunks,
            stride_per_src=M_local * N,
            CHUNK_SIZE_M=CHUNK_SIZE_M,
            SIGNAL_STRIDE=SIGNAL_STRIDE,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            RANK=rank, WORLD_SIZE=world_size,
        )

    main_stream.wait_stream(compute_stream)
    main_stream.wait_stream(reduce_stream)

    if workspace is not None:
        workspace.reset()   # zeros rs_signal_buf + arrive_count on main_stream
    else:
        rs_signal_buf.zero_()
    return output
