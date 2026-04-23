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
Dispatcher for GEMM + Reduce-Scatter.

Two backends are available for SM < 100:
  "triton"  (default) — Triton GEMM+Push kernel with NVLink peer writes and
                        inter-CTA arrive barrier.  Beats fused_matmul_reduce_scatter
                        at large M; controlled by FLASHINFER_GRS_BARRIER env var.
  "cublas"            — cuBLAS (torch.mm) + CE scatter host loop with double-
                        buffering.  Faster GEMM but higher per-call overhead;
                        see comparison.md for analysis.

SM >= 100 (Blackwell+): cuTile implementation not yet available.
"""

import os

import torch
import torch.distributed as dist

from .gemm_reduce_scatter_triton import gemm_reduce_scatter_triton
from .gemm_reduce_scatter_cublas  import gemm_reduce_scatter_cublas

def gemm_reduce_scatter(
    X_local: torch.Tensor,
    W_local: torch.Tensor,
    group: dist.ProcessGroup,
    *,
    backend: str = None,
    verbose: bool = False,
    workspace=None,   # Optional[GemmReduceScatterWorkspace]
) -> torch.Tensor:
    """
    Fused GEMM + Reduce-Scatter with overlapped computation and communication.

    Each rank holds a K-sharded slice of the input and weight:
      X_local : (M, K_local)  — K-sharded activation
      W_local : (K_local, N)  — K-sharded weight

    Returns each rank's contiguous M-slice of sum_r(X_local_r @ W_local_r):
      output : (M // world_size, N)

    Parameters
    ----------
    X_local : (M, K_local), contiguous, float16 or bfloat16.
              M must be divisible by world_size.
    W_local : (K_local, N), contiguous, same dtype as X_local.
    group   : torch.distributed.ProcessGroup
    backend : "triton" (default) or "cublas".
              Can also be set via FLASHINFER_GRS_BACKEND env var.
    verbose : If True, rank 0 prints tile/stream configuration.

    Returns
    -------
    (M // group.size(), N), same dtype, same device.
    """
    if backend is None:
        backend = os.environ.get("FLASHINFER_GRS_BACKEND", "triton").lower()

    major, _ = torch.cuda.get_device_capability(X_local.device)
    if major >= 10:
        raise NotImplementedError(
            "GEMM+Reduce-Scatter cuTile backend for SM >= 100 is not yet "
            "implemented.  Use backend='triton' or backend='cublas' on SM < 100."
        )

    if backend == "cublas":
        return gemm_reduce_scatter_cublas(X_local, W_local, group, verbose=verbose, workspace=workspace)
    elif backend == "triton":
        return gemm_reduce_scatter_triton(X_local, W_local, group, verbose=verbose, workspace=workspace)
    else:
        raise ValueError(
            f"Unknown backend {backend!r}. Choose 'triton' or 'cublas'. "
            "Can also be set via the FLASHINFER_GRS_BACKEND environment variable."
        )
