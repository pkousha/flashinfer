# cuBLAS Backend — Work In Progress

This file documents all findings, bugs, lessons learned, and the current state
of the cuBLAS GEMM+Reduce-Scatter backend.  The backend is **not yet correct**
for all configurations; it is shelved here for future work.

---

## Goal

Replace the fused Triton GEMM+push kernel with two separate pieces:
- **cuBLAS (`torch.mm`)** for the GEMM (higher arithmetic throughput on Hopper
  than a custom Triton kernel because cuBLAS uses TMA + persistent warp-groups).
- **A host CE loop** for the scatter: chunk by chunk, compute GEMM into a staging
  buffer, then push staging buffer to the destination's `recv_buf` via NVLink.

The algorithm is identical to the Triton backend (Design B, chunk-outer / dest-inner),
but the GEMM is handled by cuBLAS and the NVLink scatter is driven from the host.

---

## Architecture Recap

Both backends share:

```
recv_buf        [world_size, M_local, N]    — each rank's symmetric recv buffer
rs_signal_buf   [world_size, num_chunks, SIGNAL_STRIDE]  — one signal per (src, chunk)
_wait_reduce_kernel  — persistent tile-parallel kernel on reduce_stream that
                       spin-waits on each signal then accumulates recv_buf slices
_barrier_triton      — GPU-level all-reduce barrier using signal pads
```

The Triton backend: one fused `_gemm_push_kernel` writes GEMM tiles directly to
remote `recv_buf` slices via NVLink and fires the per-chunk signal atomically.

The cuBLAS backend (attempted): separate GEMM (cuBLAS) → staging buffer →
`copy_()` (DMA) → remote `recv_buf`; then fire signal; then reduce kernel reads.

---

## Bug 1 — cuBLAS stream deadlock (FIXED)

### Symptom
First call to `gemm_reduce_scatter_cublas` hangs forever.  Specifically the
very first `torch.mm` on `compute_stream` never returns.

### Root cause
`torch.mm` (cuBLAS) calls `cudaDeviceSynchronize()` internally on the **first
use of a new CUDA stream** to initialise its internal workspace allocator.

If the spinning `_wait_reduce_kernel` is already running on `reduce_stream` at
that point, the deadlock is:

```
cuBLAS mm on compute_stream
  → cudaDeviceSynchronize()
    → waits for ALL streams, including reduce_stream
      → reduce_stream spinning waiting for signals
        → signals come from scatter_stream
          → scatter_stream waits for compute_stream (compute_stream is blocked!)
```

### How we found it
Isolated the hang to the world_size=1 case → no NVLink involved → single-GPU
deadlock → profiled with per-step tracing → confirmed cudaDeviceSynchronize.

### Fix
Pre-warm cuBLAS on `compute_stream` with a dummy `torch.mm` of the actual CE-loop
shape **before** launching `_wait_reduce_kernel`.  Then synchronize so cuBLAS
workspace is fully initialised.  All subsequent `torch.mm` calls on that stream
are then non-blocking.

```python
_pw_a = torch.empty(CHUNK_SIZE_M, K_local, device=device, dtype=dtype)
_pw_c = torch.empty(CHUNK_SIZE_M, N,       device=device, dtype=dtype)
with torch.cuda.stream(compute_stream):
    torch.mm(_pw_a, W_local, out=_pw_c)
compute_stream.synchronize()   # block until cuBLAS initialised
del _pw_a, _pw_c               # discard garbage result
```

The reduce kernel is launched **after** this prewarm, so it never sees a
dangling `cudaDeviceSynchronize`.

---

## Bug 2 — Triton JIT compilation inside CE loop (FIXED)

### Symptom
First call hangs for 30–60 s.  Subsequent calls are fast.

### Root cause
Triton JIT **specialises on integer argument values** (0, 1, -1, powers-of-2,
etc.).  Each unique value of `chunk` in `_fire_signal_kernel(ptr, chunk, ...)`
triggers a separate compilation that **blocks the Python host** for ~10–15 s.

For `num_chunks=4` (M=8192) there are four compilations.  During compilation
the spinning reduce kernel is already running on the GPU; but Triton compilation
is CPU-side so this is harmless except for timing: Python is blocked compiling
while the CE loop cannot fire any signals.  When compilation finally finishes
the CE loop runs, signals fire, and the reduce kernel sees them.  With a 30-s
test timeout the combined compilation time can exceed the limit.

### Fix
Pre-compile `_fire_signal_kernel` for every chunk value **before** any stream
work begins:

```python
_prewarm_sig = torch.zeros(num_chunks * SIGNAL_STRIDE, device=device, dtype=Configs.SIGNAL_DTYPE)
for _c in range(num_chunks):
    _fire_signal_kernel[(1,)](_prewarm_sig, _c, SIGNAL_STRIDE=SIGNAL_STRIDE)
del _prewarm_sig
```

This is a host-side warm-up call; the dummy kernel runs on the current (main)
stream.  After this the Triton cache has all needed specialisations and
subsequent calls are instant.

---

## Bug 3 — NVLink DMA write visibility (UNSOLVED — root cause of remaining failures)

### Symptom
For `M=8192` (num_chunks=4), `_wait_reduce_kernel` produces **wrong results**
(max_abs_err ≈ 400–700, atol=3.0).  The error is large and consistent — not a
numerical precision issue, but missing or stale data.  For `M=4096` (num_chunks=2)
results are correct (max_err ≈ 2.0).

### Root cause
`copy_(buf, non_blocking=True)` uses CUDA's **DMA Copy Engine**, not SM writes.

CUDA's PTX memory model: `atom.release.sys` (from `_fire_signal_kernel`) fences
all **SM-level writes** that happened before it.  DMA engine writes bypass the SM's
cache hierarchy and are **not included** in this release fence.

The receiving GPU's `_wait_reduce_kernel` polls the signal with
`tl.load(sig_ptr, volatile=True)`.  Volatile load reads from L2 (bypasses L1).
When the volatile load sees `signal=1` the SM release has fired — but the
DMA data may not have propagated to the receiver's L2 yet.

Timeline of the race:
```
Rank 1, scatter_stream (FIFO):
  copy_(stage_buf → rank0.recv_buf[1, rows])  ← DMA write to rank0 L2
  _fire_signal_kernel → rank0.rs_signal_buf[1, chunk] = 1  ← SM atomic

Rank 0, reduce_stream:
  volatile load rs_signal_buf[1,chunk]  → sees 1 (signal arrived)
  tl.load(recv_buf[1, rows])            → may see STALE data (DMA not in L2 yet)
```

The CUDA SM-scope release/acquire chain does **not** cover DMA writes from a
different GPU's Copy Engine.

### Why M=4096 works but M=8192 sometimes fails
With num_chunks=2 the DMA copy for each chunk is smaller and completes faster
relative to the time it takes for the signal to traverse NVLink.  By the time
the signal is observed by the reduce kernel, the DMA data has usually landed
in L2.  With num_chunks=4 the race window is wider and the failure rate rises.

### Approaches tried and why they failed

**Attempt A — `_scatter_and_signal_kernel` (SM tl.store instead of DMA)**

Replaced `copy_() + _fire_signal_kernel` with a single Triton kernel that copies
data using `tl.store` (SM writes) then fires the signal with `sem="release"`.
SM writes ARE covered by the release fence, so the acquire/release pairing
should formally guarantee visibility.

Why it failed: **stage buffer race**.  In the CE loop, `stage_bufs[dest_shift]`
is reused across chunks.  When `_scatter_and_signal_kernel` copies from
`stage_bufs[0]` for `chunk=0` (slow, sequential 512-iteration loop over 2 MB),
the GEMM for `chunk=2` (which also uses `stage_bufs[0]`) has already started on
`compute_stream` and overwrites the buffer.  `copy_()` (DMA, ~2 µs) is fast enough
that this window is negligible; a 512-iteration SM loop (~100–500 µs) is not.

The formal acquire/release correctness argument was sound but irrelevant because
corrupted source data was being sent.

**Attempt B — second `_barrier_triton` after CE loop**

Added a second GPU-level barrier after the CE loop so both ranks know each
other's scatter streams are complete before either launches its reduce kernel.

Why it insufficient: `_barrier_triton` uses SM atomics (release/acquire).
Like Bug 3 above, the SM-level release of the barrier deposit does not formally
cover the DMA writes from the CE loop.  In practice it sometimes helps because
the DMA writes typically arrive in peer L2 before the barrier completes — but
this is a timing argument, not a formal guarantee, and empirically fails ~33% of
runs for M=8192.

**Attempt C — changing `_wait_reduce_kernel` to use `_acquire_load`**

Changed the signal spin-wait from `tl.load(sig, volatile=True)` to the PTX
`ld.global.acquire.sys.b32` inline asm.  This broke the **Triton backend** because
the `_acquire_load` function (used in the inter-CTA arrive barrier) apparently
behaves differently inside the spin-wait context — the Triton backend hung and
produced wrong results.  Reverted immediately.

---

## Bug 4 — Stage buffer reuse race (UNSOLVED)

### Symptom
When using a slow scatter (SM copies, ~100–500 µs per chunk), the GEMM for
chunk `c+1` overwrites `stage_bufs[dest_shift]` while the scatter for chunk `c`
is still reading from the same buffer.

### Root cause
In the CE loop each staging buffer `stage_bufs[dest_shift]` (indexed 0…ws-1)
is reused on every chunk.  For chunk 0 and chunk 2 both use `stage_bufs[0]`.
GEMM runs on `compute_stream`; scatter runs on `scatter_stream`; `scatter_stream`
waits for `compute_stream` but NOT vice versa.  So the GEMM for chunk 2 can
start while scatter for chunk 0 is still running on a different stream.

With `copy_()` (DMA, completes in ~2 µs for 2 MB at 900 GB/s NVLink), the
scatter for chunk 0 finishes long before the GEMM for chunk 2 starts.  With an
SM-based Triton copy kernel (~200–500 µs for the same data), the race is wide.

### Potential fixes (not implemented)
1. Add `compute_stream.wait_stream(scatter_stream)` inside the inner loop so
   the next GEMM does not start until the previous scatter completes.  This
   serialises GEMM and scatter completely, removing the race.
2. Allocate separate staging buffers per `(chunk, dest_shift)` pair — i.e.,
   `num_chunks * world_size` buffers.  This eliminates the race without
   serialisation.  Memory cost: `num_chunks * world_size * CHUNK_SIZE_M * N * sizeof(dtype)`
   = e.g. `4 * 2 * 1024 * 2048 * 2 ≈ 32 MB` for M=8192, ws=2.  Acceptable.

---

## Correct Ordering for the cuBLAS Backend

The following ordering is what `test_m8192_correct.py` showed to be reliable
when the Triton kernel cache is pre-warmed and both DMA copies and signals are
fast (cache-warm run):

```
Rank 0 and Rank 1 (concurrent):
  1. _barrier_triton  (initial sync)
  2. Prewarm signal kernel (compile)
  3. Prewarm cuBLAS (compile + init)
  4. CE loop:
       for chunk in 0..num_chunks-1:
         for dest_shift in 0..ws-1:
           GEMM → stage_bufs[dest_shift]          (compute_stream)
           scatter_stream.wait_stream(compute_stream)
           copy_(stage_bufs[dest_shift] → remote_recv_ptrs[dest_shift][chunk_rows])  (DMA)
           _fire_signal_kernel(signal, chunk)      (scatter_stream)
  5. main_stream.wait_stream(scatter_stream)    (GPU event, not host-blocking)
  6. _barrier_triton  (ensures BOTH ranks' DMA writes landed in peer L2)
  7. reduce_stream.wait_stream(main_stream)
  8. _wait_reduce_kernel  (signals all 1 from own scatter; spins for remote)
  9. main_stream.wait_stream(reduce_stream)
 10. rs_signal_buf.zero_()  / workspace.reset()
```

The second barrier (step 6) ensures DMA writes from both ranks are in peer L2
before the reduce kernel reads them.  This is empirically reliable but not
formally guaranteed by CUDA's memory model (DMA not covered by SM release/acquire).

---

## Current State of the Code

The file `gemm_reduce_scatter_cublas.py` in this directory has:
- Bug 1 (cuBLAS deadlock) FIXED via compute_stream prewarm
- Bug 2 (Triton JIT blocking) FIXED via signal kernel prewarm
- Bug 3 (NVLink DMA visibility) partially addressed with second barrier
- Bug 4 (stage buffer race) NOT present with DMA (fast enough)
- Still intermittently wrong for M=8192 (~33% failure rate)

The Triton backend (`gemm_reduce_scatter_triton.py`) is unaffected and correct.

---

## Lessons Learned

1. **cuBLAS stream initialisation**: `torch.mm` on a fresh CUDA stream
   calls `cudaDeviceSynchronize()` once to set up its workspace.  Never launch
   a spinning GPU kernel before the first `torch.mm` on a stream.

2. **DMA vs SM writes for NVLink**: CUDA's DMA Copy Engine writes bypass the
   SM cache hierarchy.  SM-level release fences (`atom.release.sys`) do NOT
   cover DMA writes.  If correctness depends on the receiver seeing data that
   was written via DMA, you need either:
   - A `membar.sys` PTX instruction after the DMA completes (not accessible from Triton),
   - SM-based writes (`tl.store`) for the data transfer,
   - A full distributed barrier (`dist.barrier()` via NCCL), or
   - Launching the reader AFTER a CUDA stream event that implies the DMA is done.

3. **Stage buffer reuse race**: In a CE loop that reuses output buffers, ensure
   the scatter stream that reads a buffer has fully completed before the compute
   stream writes to the same buffer for the next chunk.  DMA is fast enough to
   avoid this in practice; a slow Triton copy kernel is not.

4. **Triton JIT specialisation on runtime integers**: Each unique integer value
   passed to a Triton kernel as a non-constexpr argument may trigger a separate
   compilation.  Pre-warm all values before latency-sensitive code paths,
   especially before launching spinning kernels.

5. **`_acquire_load` in `_wait_reduce_kernel`**: Replacing the volatile load with
   `ld.global.acquire.sys.b32` in the signal spin-wait broke the Triton backend.
   The exact mechanism is not fully understood; the volatile load with a
   `sem="release"` sender is sufficient in practice for the Triton backend which
   uses SM stores (not DMA) for data writes.  Do not change this without careful
   testing of both backends.

6. **Two-barrier pattern**: Placing `_barrier_triton` at the START of the function
   (already done) and at the END (after CE loop) partially fixes NVLink visibility
   for DMA.  But it is not formally correct because the barrier uses SM atomics,
   not a system-scope membar.  It works empirically ~67% of the time for M=8192.

---

## Recommended Path Forward

The cleanest fix that avoids ALL three remaining issues:

**Use SM-based scatter + per-(chunk, dest_shift) staging buffers:**

```
stage_bufs = [
    torch.empty(CHUNK_SIZE_M, N, device=device, dtype=dtype)
    for _ in range(num_chunks * world_size)
]
```

Then in the CE loop, `buf = stage_bufs[chunk * world_size + dest_shift]` — each
(chunk, dest_shift) pair gets its own buffer.  No reuse race.

For the scatter, use `_scatter_and_signal_kernel` (SM tl.store + release atomic)
so the NVLink data write IS covered by the release fence.  Use `volatile=True`
in the receiver's spin-wait (proven to work with SM-sender + SM-release).

The reduce kernel can be launched BEFORE the CE loop (spinning), which gives
better overlap.  The DMA visibility issue doesn't exist because we're using SM
stores.

Memory overhead: `num_chunks * world_size * CHUNK_SIZE_M * N * 2 bytes`.
For M=65536 (max tested), ws=8: `64 * 8 * 1024 * 2048 * 2 = 2 GB`.  Too large.
For M=8192, ws=2: `4 * 2 * 1024 * 2048 * 2 = 32 MB`.  Acceptable.
Use workspace to pre-allocate for a fixed M.

Alternatively, serialise scatter after each GEMM (simpler but lower throughput):
add `compute_stream.wait_stream(scatter_stream)` after each scatter kernel launch.
