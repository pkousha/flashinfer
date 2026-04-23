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
GemmReduceScatterWorkspace — pre-allocated resource container.

Eliminates the three symm_mem.rendezvous() collectives and two torch.cuda.Stream()
creations that currently happen on every call to gemm_reduce_scatter_triton/cublas.

Without workspace (current):
  Every call: symm_mem.empty×3 + rendezvous×3 + Stream()×2 + zeros×1
  rendezvous is a collective — all ranks must participate synchronously.
  Overhead: ~4–5 ms per call, dominates latency at small M.

With workspace (this class):
  One-time setup: same allocations done in __init__
  Per call: zero signal buffers + arrive_count only (~few µs)
  Per call overhead: negligible.

Design:
  - Fixed M at creation time (workspace sized for one specific M value).
  - Both Triton and cuBLAS backends share the same workspace.
  - Provides reset() for signal/counter cleanup between calls.
  - Context-manager support (__enter__/__exit__) for safe cleanup.

Usage:
    ws = GemmReduceScatterWorkspace(
        M=16384, N=2048, K_local=2048,
        group=group, dtype=torch.bfloat16, device=device,
    )
    for step in range(1000):
        out = gemm_reduce_scatter(X, W, group, workspace=ws)
    ws.destroy()

    # Or as context manager:
    with GemmReduceScatterWorkspace(...) as ws:
        out = gemm_reduce_scatter(X, W, group, workspace=ws)
"""

import warnings

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton

from .configs import Configs
from .gemm_reduce_scatter_triton import (
    _BLOCK_M, _BLOCK_N, _CHUNK_SIZE_M, _SIGNAL_STRIDE, _ARRIVE_STRIDE,
)


class GemmReduceScatterWorkspace:
    """
    Pre-allocated workspace for gemm_reduce_scatter.

    All symm_mem tensors, IPC handles, remote pointers, CUDA streams, and
    staging buffers are allocated once in __init__ and reused across calls.

    Parameters
    ----------
    M        : total sequence length (X_local.shape[0])
    N        : output dimension
    K_local  : local K dimension (K // world_size)
    group    : torch.distributed.ProcessGroup
    dtype    : bfloat16 or float16
    device   : CUDA device

    The workspace is valid for the exact (M, N, K_local, world_size, dtype) it
    was created with.  Call is_compatible() to verify before use.
    """

    def __init__(
        self,
        M: int,
        N: int,
        K_local: int,
        group: dist.ProcessGroup,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        Configs.initialize()

        self._destroyed = False

        world_size = dist.get_world_size(group)
        rank       = dist.get_rank(group)

        assert M % world_size == 0, f"M ({M}) must be divisible by world_size ({world_size})"
        M_local = M // world_size

        CHUNK_SIZE_M  = _CHUNK_SIZE_M
        SIGNAL_STRIDE = _SIGNAL_STRIDE
        ARRIVE_STRIDE = _ARRIVE_STRIDE

        assert M_local % CHUNK_SIZE_M == 0, (
            f"M_local ({M_local}) must be divisible by CHUNK_SIZE_M ({CHUNK_SIZE_M})"
        )
        num_chunks = M_local // CHUNK_SIZE_M

        # Store dimensions for is_compatible() checks
        self.M          = M
        self.N          = N
        self.K_local    = K_local
        self.M_local    = M_local
        self.num_chunks = num_chunks
        self.world_size = world_size
        self.rank       = rank
        self.dtype      = dtype
        self.device     = device

        # ------------------------------------------------------------------
        # Symmetric memory — three separate tensors (see design.md §6).
        # ------------------------------------------------------------------

        # 1. Barrier buffer (tiny, barrier signals only)
        self.barrier_buf    = symm_mem.empty(world_size, device=device, dtype=Configs.SIGNAL_DTYPE)
        self._barrier_handle = symm_mem.rendezvous(self.barrier_buf, group.group_name)
        self.barrier_sigpads = tuple(
            self._barrier_handle.get_signal_pad(r, (world_size,), Configs.SIGNAL_DTYPE, 0)
            for r in range(world_size)
        )

        # 2. Data recv buffer
        self.recv_buf = symm_mem.empty(world_size, M_local, N, device=device, dtype=dtype)
        self._recv_handle = symm_mem.rendezvous(self.recv_buf, group.group_name)

        # 3. Per-chunk GEMM+RS signal buffer (cache-line padded)
        self.rs_signal_buf = symm_mem.empty(
            world_size, num_chunks, SIGNAL_STRIDE,
            device=device, dtype=Configs.SIGNAL_DTYPE,
        )
        self._rs_signal_handle = symm_mem.rendezvous(self.rs_signal_buf, group.group_name)

        # ------------------------------------------------------------------
        # Pre-computed remote pointers (indexed by dest_shift = 0..ws-1)
        # ------------------------------------------------------------------
        self.remote_recv_ptrs = tuple(
            self._recv_handle.get_buffer(
                (rank + shift) % world_size,
                (world_size, M_local, N),
                dtype,
            )[rank]
            for shift in range(world_size)
        )

        self.remote_signal_ptrs = tuple(
            self._rs_signal_handle.get_buffer(
                (rank + shift) % world_size,
                (world_size, num_chunks, SIGNAL_STRIDE),
                Configs.SIGNAL_DTYPE,
            )[rank]
            for shift in range(world_size)
        )

        # ------------------------------------------------------------------
        # arrive_count for the Triton inter-CTA barrier
        # Zeroed at reset() after each call.
        # ------------------------------------------------------------------
        self.arrive_count = torch.zeros(
            num_chunks * world_size * ARRIVE_STRIDE,
            dtype=torch.int32,
            device=device,
        )

        # ------------------------------------------------------------------
        # CUDA streams — created once, reused across calls
        # ------------------------------------------------------------------
        self.compute_stream = torch.cuda.Stream(device=device)  # Triton GEMM+Push / cuBLAS mm
        self.scatter_stream = torch.cuda.Stream(device=device)  # cuBLAS CE scatter only
        self.reduce_stream  = torch.cuda.Stream(device=device)  # Wait+Reduce (both backends)

        # ------------------------------------------------------------------
        # Staging buffers for cuBLAS backend (world_size × (CHUNK_SIZE_M, N))
        # ------------------------------------------------------------------
        self.stage_bufs = [
            torch.empty(CHUNK_SIZE_M, N, device=device, dtype=dtype)
            for _ in range(world_size)
        ]

        # ------------------------------------------------------------------
        # SM configuration for Wait+Reduce (fixed at creation)
        # ------------------------------------------------------------------
        BLOCK_M = _BLOCK_M
        BLOCK_N = _BLOCK_N
        NUM_SMS = torch.cuda.get_device_properties(device).multi_processor_count
        reserve_reduce    = max(1, NUM_SMS // 4)
        total_out_tiles   = triton.cdiv(M_local, BLOCK_M) * triton.cdiv(N, BLOCK_N)
        self.reduce_grid  = (min(reserve_reduce, total_out_tiles),)
        self.NUM_SMS_compute = NUM_SMS - reserve_reduce

        # Store kernel kwargs that are fixed per workspace
        self.CHUNK_SIZE_M  = CHUNK_SIZE_M
        self.SIGNAL_STRIDE = SIGNAL_STRIDE
        self.ARRIVE_STRIDE = ARRIVE_STRIDE

    # ------------------------------------------------------------------
    # Per-call lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Zero signal and arrive_count buffers between calls.

        Must be called on main_stream AFTER both compute/reduce streams have
        joined — i.e., after main_stream.wait_stream(reduce_stream) completes.
        Calling reset() before that risks zeroing buffers still being read.

        Both backends call this internally at the end of each forward pass.
        """
        self.rs_signal_buf.zero_()
        self.arrive_count.zero_()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def is_compatible(
        self,
        M: int,
        N: int,
        K_local: int,
        world_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> bool:
        """Return True if this workspace can handle the given problem dimensions.

        All six dimensions must match — including device.  A workspace created
        on GPU 0 must not be reused on GPU 1.
        """
        return (
            M == self.M
            and N == self.N
            and K_local == self.K_local
            and world_size == self.world_size
            and dtype == self.dtype
            and device == self.device
        )

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def destroy(self) -> None:
        """
        Explicitly free symm_mem resources.

        Should be called when the workspace is no longer needed.  If not called,
        __del__ will attempt cleanup but will emit a ResourceWarning.
        """
        if self._destroyed:
            return

        # Deleting the symm_mem tensors releases the IPC handles and
        # frees the symmetric heap allocations.
        del self.barrier_buf
        del self._barrier_handle
        del self.recv_buf
        del self._recv_handle
        del self.rs_signal_buf
        del self._rs_signal_handle
        del self.arrive_count
        del self.stage_bufs
        del self.compute_stream
        del self.scatter_stream
        del self.reduce_stream

        self._destroyed = True

    def __del__(self) -> None:
        if not self._destroyed:
            warnings.warn(
                "GemmReduceScatterWorkspace was not explicitly destroyed. "
                "Call workspace.destroy() or use it as a context manager to "
                "ensure proper cleanup of distributed/CUDA resources.",
                ResourceWarning,
                stacklevel=2,
            )
            try:
                self.destroy()
            except Exception as e:
                warnings.warn(
                    f"Error during automatic cleanup of GemmReduceScatterWorkspace: {e}",
                    ResourceWarning,
                    stacklevel=2,
                )

    # ------------------------------------------------------------------
    # Thread safety note
    # ------------------------------------------------------------------
    # This workspace is NOT thread-safe for concurrent calls.  All resources
    # (streams, signal buffers, arrive_count) are shared.  Concurrent Python
    # threads calling gemm_reduce_scatter(..., workspace=ws) will race on
    # rs_signal_buf and arrive_count.  Use one workspace per thread or
    # protect with a lock.

    def __enter__(self) -> "GemmReduceScatterWorkspace":
        return self

    def __exit__(self, *args) -> None:
        self.destroy()

    def __repr__(self) -> str:
        return (
            f"GemmReduceScatterWorkspace("
            f"M={self.M}, N={self.N}, K_local={self.K_local}, "
            f"ws={self.world_size}, dtype={self.dtype}, "
            f"device={self.device}, "
            f"destroyed={self._destroyed})"
        )
