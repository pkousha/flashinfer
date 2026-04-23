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

"""Shared configuration for GEMM + Reduce-Scatter kernels."""

import os

import torch


class Configs:
    # -----------------------------------------------------------------------
    # Barrier mode
    # -----------------------------------------------------------------------
    # "spin"    — volatile spin-wait (all SM generations, default)
    # "mbarrier" — ld.global.acquire.sys spin-wait (SM >= 90 / Hopper only)
    # Controlled by FLASHINFER_GRS_BARRIER env var.
    BARRIER_MODE: str = "spin"

    # -----------------------------------------------------------------------
    # Signal dtype
    # -----------------------------------------------------------------------
    # uint32 signals: 0 = not ready, 1 = data written and visible.
    SIGNAL_DTYPE: torch.dtype = torch.uint32
    SIGNAL_ELEM_BYTES: int = 4  # sizeof(uint32)

    # -----------------------------------------------------------------------
    # Cache line padding
    # -----------------------------------------------------------------------
    # NVIDIA GPU L2 cache lines are 128 bytes.  With uint32 (4 bytes) elements,
    # one cache line holds 32 elements.
    #
    # rs_signal (per-chunk ready signals) is padded so each logical slot
    # occupies exactly one cache line.  This eliminates two sources of L2
    # pressure:
    #
    #   1. False sharing: adjacent slots in a packed layout share a cache line.
    #      When one slot is updated, the hardware must invalidate the entire
    #      line on all other SMs — even those spinning on a different slot.
    #
    #   2. Hot-line thundering herd: all spinning CTAs on the same 4-byte
    #      location cause repeated cache-line invalidation and re-fetch storms
    #      when the value changes.  Isolating each slot onto its own line
    #      limits invalidation to the CTAs actually spinning on that slot.
    #
    # See design.md §7 for the full cache footprint analysis.
    CACHE_LINE_WORDS: int = 32  # 128 bytes / 4 bytes per uint32

    initialized: bool = False

    @classmethod
    def initialize(cls) -> None:
        if cls.initialized:
            return
        cls.initialized = True
        barrier_mode = os.environ.get("FLASHINFER_GRS_BARRIER", "spin").lower()
        if barrier_mode not in ("spin", "mbarrier"):
            raise ValueError(
                f"FLASHINFER_GRS_BARRIER={barrier_mode!r} is invalid. "
                "Choose 'spin' or 'mbarrier'."
            )
        cls.BARRIER_MODE = barrier_mode
