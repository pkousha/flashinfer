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
cuBLAS (torch.mm) + CE scatter implementation of GEMM + Reduce-Scatter.

STATUS: WORK IN PROGRESS — correctness not guaranteed for all configurations.
See CUBLAS_WIP.md in this directory for the full list of known bugs and the
recommended path forward.

Known issues:
  - NVLink DMA visibility: copy_() writes are not covered by SM release fences;
    the receiver may see the signal before the data is in its L2 cache.
  - Stage buffer race: with SM-based scatter (slow), GEMM for chunk c+1 can
    overwrite stage_bufs[dest_shift] while scatter for chunk c is still reading.

The Triton backend (gemm_reduce_scatter_triton.py) is stable and correct.
"""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

from .configs import Configs
from .gemm_reduce_scatter_triton import (
    _barrier_triton,
    _wait_reduce_kernel,
    _BLOCK_M,
    _BLOCK_N,
    _CHUNK_SIZE_M,
    _SIGNAL_STRIDE,
)


# ---------------------------------------------------------------------------
# Signal kernel
# ---------------------------------------------------------------------------

@triton.jit
def _fire_signal_kernel(signal_ptr, chunk, SIGNAL_STRIDE: tl.constexpr):
    """
    Write rs_signal[rank, chunk] = 1 with release semantics.  Grid = (1,).

    NOTE: This uses an SM-level release fence which does NOT cover preceding
    DMA (copy_()) writes to peer NVLink memory.  See CUBLAS_WIP.md §Bug 3.
    """
    if tl.program_id(0) == 0:
        tl.atomic_xchg(signal_ptr + chunk * SIGNAL_STRIDE, 1, sem="release")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def gemm_reduce_scatter_cublas(
    X_local: torch.Tensor,
    W_local: torch.Tensor,
    group: dist.ProcessGroup,
    *,
    verbose: bool = False,
    workspace=None,   # Optional[GemmReduceScatterWorkspace]
) -> torch.Tensor:
    """
    cuBLAS GEMM + Reduce-Scatter.  Work-in-progress — see CUBLAS_WIP.md.

    Parameters
    ----------
    X_local : (M, K_local) bf16/fp16, contiguous.
    W_local : (K_local, N) same dtype, contiguous.
    group   : process group.

    Returns
    -------
    (M_local, N) — approximate (may have correctness issues for large M).
    """
    Configs.initialize()

    assert X_local.is_contiguous(), "X_local must be contiguous"
    assert W_local.is_contiguous(), "W_local must be contiguous"

    M, K_local = X_local.shape
    K_w,     N = W_local.shape
    assert K_w == K_local

    world_size = dist.get_world_size(group)
    rank       = dist.get_rank(group)

    assert M % world_size == 0
    M_local = M // world_size

    CHUNK_SIZE_M  = _CHUNK_SIZE_M
    SIGNAL_STRIDE = _SIGNAL_STRIDE

    assert M_local % CHUNK_SIZE_M == 0
    num_chunks = M_local // CHUNK_SIZE_M

    device = X_local.device
    dtype  = X_local.dtype

    # ------------------------------------------------------------------
    # Symmetric memory
    # ------------------------------------------------------------------
    if workspace is not None:
        assert workspace.is_compatible(M, N, K_local, world_size, dtype, device)
        barrier_sigpads    = workspace.barrier_sigpads
        recv_buf           = workspace.recv_buf
        rs_signal_buf      = workspace.rs_signal_buf
        remote_recv_ptrs   = workspace.remote_recv_ptrs
        remote_signal_ptrs = workspace.remote_signal_ptrs
        stage_bufs         = workspace.stage_bufs
    else:
        barrier_buf    = symm_mem.empty(world_size, device=device, dtype=Configs.SIGNAL_DTYPE)
        barrier_handle = symm_mem.rendezvous(barrier_buf, group.group_name)
        barrier_sigpads = tuple(
            barrier_handle.get_signal_pad(r, (world_size,), Configs.SIGNAL_DTYPE, 0)
            for r in range(world_size)
        )
        recv_buf = symm_mem.empty(world_size, M_local, N, device=device, dtype=dtype)
        handle   = symm_mem.rendezvous(recv_buf, group.group_name)
        rs_signal_buf = symm_mem.empty(
            world_size, num_chunks, SIGNAL_STRIDE,
            device=device, dtype=Configs.SIGNAL_DTYPE,
        )
        rs_signal_handle = symm_mem.rendezvous(rs_signal_buf, group.group_name)
        remote_recv_ptrs = tuple(
            handle.get_buffer((rank + s) % world_size, (world_size, M_local, N), dtype)[rank]
            for s in range(world_size)
        )
        remote_signal_ptrs = tuple(
            rs_signal_handle.get_buffer(
                (rank + s) % world_size, (world_size, num_chunks, SIGNAL_STRIDE),
                Configs.SIGNAL_DTYPE,
            )[rank]
            for s in range(world_size)
        )
        stage_bufs = None

    # ------------------------------------------------------------------
    # SM config
    # ------------------------------------------------------------------
    BLOCK_M = _BLOCK_M
    BLOCK_N = _BLOCK_N
    NUM_SMS = torch.cuda.get_device_properties(device).multi_processor_count
    reserve_reduce = max(1, NUM_SMS // 4)
    total_out_tiles = triton.cdiv(M_local, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    reduce_grid = (min(reserve_reduce, total_out_tiles),)

    main_stream = torch.cuda.current_stream()

    if verbose and rank == 0:
        print(
            f"[gemm_reduce_scatter_cublas] M={M} K_local={K_local} N={N} "
            f"ws={world_size} M_local={M_local} num_chunks={num_chunks} "
            f"CHUNK={CHUNK_SIZE_M} BLOCK=({BLOCK_M},{BLOCK_N}) "
            f"reduce_SMs={reserve_reduce} backend=cublas"
        )

    # ------------------------------------------------------------------
    # Barrier (initial cross-rank sync)
    # ------------------------------------------------------------------
    _barrier_triton(barrier_sigpads, rank)

    # ------------------------------------------------------------------
    # Pre-warm _fire_signal_kernel for all chunk values.
    # Triton specialises on integer values; pre-compiling here avoids
    # blocking the host inside the CE loop (see CUBLAS_WIP.md §Bug 2).
    # ------------------------------------------------------------------
    _prewarm_sig = torch.zeros(num_chunks * SIGNAL_STRIDE, device=device, dtype=Configs.SIGNAL_DTYPE)
    for _c in range(num_chunks):
        _fire_signal_kernel[(1,)](_prewarm_sig, _c, SIGNAL_STRIDE=SIGNAL_STRIDE)
    del _prewarm_sig

    # ------------------------------------------------------------------
    # Streams + staging buffers + cuBLAS warm-up.
    # cuBLAS calls cudaDeviceSynchronize() on first stream use.
    # Must warm up BEFORE launching any spinning kernel (see CUBLAS_WIP.md §Bug 1).
    # ------------------------------------------------------------------
    if workspace is not None:
        compute_stream = workspace.compute_stream
        scatter_stream = workspace.scatter_stream
    else:
        compute_stream = torch.cuda.Stream()
        scatter_stream = torch.cuda.Stream()
        stage_bufs = [
            torch.empty(CHUNK_SIZE_M, N, device=device, dtype=dtype)
            for _ in range(world_size)
        ]

    _pw_a = torch.empty(CHUNK_SIZE_M, K_local, device=device, dtype=dtype)
    _pw_c = torch.empty(CHUNK_SIZE_M, N,       device=device, dtype=dtype)
    with torch.cuda.stream(compute_stream):
        torch.mm(_pw_a, W_local, out=_pw_c)
    compute_stream.synchronize()
    del _pw_a, _pw_c

    # ------------------------------------------------------------------
    # CE loop
    # ------------------------------------------------------------------
    compute_stream.wait_stream(main_stream)
    scatter_stream.wait_stream(main_stream)

    output = torch.empty(M_local, N, device=device, dtype=dtype)

    for chunk in range(num_chunks):
        for dest_shift in range(world_size):
            buf  = stage_bufs[dest_shift]
            dest = (rank + dest_shift) % world_size

            x_row_start = dest * M_local + chunk * CHUNK_SIZE_M
            X_chunk = X_local[x_row_start : x_row_start + CHUNK_SIZE_M]

            with torch.cuda.stream(compute_stream):
                torch.mm(X_chunk, W_local, out=buf)

            scatter_stream.wait_stream(compute_stream)
            with torch.cuda.stream(scatter_stream):
                remote_recv_ptrs[dest_shift][
                    chunk * CHUNK_SIZE_M : (chunk + 1) * CHUNK_SIZE_M
                ].copy_(buf, non_blocking=True)
                _fire_signal_kernel[(1,)](
                    remote_signal_ptrs[dest_shift], chunk, SIGNAL_STRIDE=SIGNAL_STRIDE,
                )

    # ------------------------------------------------------------------
    # Cross-rank barrier after CE loop.
    # Partially mitigates Bug 3 (NVLink DMA visibility): when both ranks
    # have passed this barrier, their scatter_streams are complete and
    # DMA writes have typically arrived in peer L2.  Not a formal guarantee.
    # See CUBLAS_WIP.md §Bug 3 for the full analysis.
    # ------------------------------------------------------------------
    main_stream.wait_stream(scatter_stream)
    _barrier_triton(barrier_sigpads, rank)

    # ------------------------------------------------------------------
    # Wait+Reduce after barrier
    # ------------------------------------------------------------------
    reduce_stream = workspace.reduce_stream if workspace is not None else torch.cuda.Stream()
    reduce_stream.wait_stream(main_stream)

    with torch.cuda.stream(reduce_stream):
        _wait_reduce_kernel[reduce_grid](
            recv_buf, rs_signal_buf, output,
            M_local=M_local, N=N, num_chunks=num_chunks,
            stride_per_src=M_local * N,
            CHUNK_SIZE_M=CHUNK_SIZE_M, SIGNAL_STRIDE=SIGNAL_STRIDE,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            RANK=rank, WORLD_SIZE=world_size,
        )

    main_stream.wait_stream(reduce_stream)

    if workspace is not None:
        workspace.reset()
    else:
        rs_signal_buf.zero_()
    return output
