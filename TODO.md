# TODO

What still needs doing in the `birman` fork, as of 2026-10-05. Everything else we found has been fixed and is
described in the notice at the top of `README.md`.

## Code gaps

- [ ] **No NEON I2_S kernel.** On ARM the I2_S mat-mul runs the scalar fallback. Since patch `0007` it is correct
  (layout, `sum(code*y)` convention and row tails all match the AVX2 path, and it was verified on an x86 build with
  AVX2 off), but it is slow. A NEON `vec_dot`/`gemm` (with the row tail from `ggml_i2s_tail_dot`) would be a
  performance item, and the harness approach used for `0006`/`0007` (random ternary matrices at several row
  lengths against a scalar reference, now `utils/test_i2s_kernels.c`) would verify it. `src/ggml-bitnet-mad.cpp`
  has old NEON kernels, but it is not compiled into the build and uses a 64-element block that does not match the
  128-element packing.
- [ ] **The `*_interleaved` GEMM/GEMV functions have no callers.** They now assert on rows that are not a multiple
  of 128 instead of silently skipping the tail. Delete them or give them tail support if something starts using them.
- [ ] Row lengths for I2_S must be a multiple of 4 (asserted in `quantize_i2_s`).

## ARM / Apple Silicon: first steps

Nothing has run on real ARM yet (see `README.md`, Known issues). The compiled I2_S kernels on aarch64 are the scalar
fallback only; `ggml-aarch64.c` and `ggml-bitnet-compute.c` in the submodule are never built. `CLAUDE.md` (gitignored,
copy it by hand) has the full handoff; the essentials:

- [ ] `./build.sh`, then the kernel unit test (`utils/test_i2s_kernels.c`, must print `ALL OK`), then a 2B-4T generation
  (`The capital of France is Paris. Paris is a city that is known for its rich history, culture,`) and
  `python utils/test_server_api.py` (18/18).
- [ ] I2_S against f32 perplexity on `bitnet_b1_58-large` (WikiText-2 100 k slice; reference 12.9532 against 12.9638 on
  8 chunks, 11.8181 on the whole slice), then speed with `llama-bench` and the performance-core count as `-t`.
- [ ] A NEON `ggml_vec_dot_i2_i8_s` and `ggml_gemm_i2_i8_s` (new patch `0009`), following the layout, the
  `sum(code*y)` convention and the row tail in `ggml_i2s_tail_dot`; verify with the unit test, then perplexity.
- [ ] macOS portability: `start_llama.py` reads `/proc` and `/sys` (thread default would count efficiency cores; use
  `sysctl hw.perflevel0.physicalcpu`, `hw.memsize`, `vm_stat`), `build.sh` assumes bash 4+, `utils/test_gemm_kernel.sh`
  looks for `libggml.so` and uses `-march=native`, `utils/test_power.sh` is Intel/Linux only, and
  `utils/cleanup_stale_models.sh` uses GNU `stat`/`numfmt`.
- [ ] TL1: `setup_env.py` runs `codegen_tl1.py` on arm64 but always passes `-DBITNET_ARM_TL1=OFF`, so TL1 is never
  compiled in; untested.

## NUMA

- [ ] `--numa distribute` is only verified on one 2-socket, 2-node Xeon (E5-2682 v4). Machines with more NUMA nodes
  (for example AMD EPYC, which can expose 4-8 nodes), and ARM servers, are untested. The default applies whenever
  Linux reports more than one node with CPUs (`numa_distribute.py`), so check it there before relying on it.
- [ ] The gain depends on the pinned threads first-touching the model's pages. After a copy or download the pages
  sit on one node and a run gains only about 10-15%; today the user has to evict the file (`--numa-evict`) once.
  Detecting a misplaced cache cheaply (per-file NUMA placement is not exposed without `move_pages` or
  `/proc/<pid>/numa_maps` of a running process) and evicting automatically only then would remove that step.
- [ ] `bitnet_b1_58-large` prompt processing at 16 threads was 0.71x slower with `--numa distribute` (466 against
  660 t/s). Every other case was within 10% or faster; not investigated.
- [ ] `start_llama.py` defaults `-t` to the physical core count, which maximises generation; prompt-heavy workloads
  gained a little from SMT threads on the Xeon (2B-4T pp128 516 at 64 threads against 416 at 32) and almost nothing
  on the i9 (209 against 205). A `--prompt-heavy` option, or choosing by workload, is untested. On non-Linux systems
  the thread count falls back to the logical CPU count, and only AVX2/ARM warnings are tuned for x86_64 and aarch64.
- [ ] `run_inference_server.py` was only smoke-tested with the flag (starts, answers, 44.5 t/s on 2B-4T), not
  benchmarked under load.

## Build

- [ ] OpenMP built with clang against GCC's `libgomp` was about 10x slower on the Xeon (2B-4T pp512 18.8 t/s against
  470); not investigated. The default build requests OpenMP, but with no `libomp` installed CMake silently builds
  without it, which is the fast configuration. LTO was mixed (+11% prompt on 2B-4T, -8% on Llama3-8B), so it was not adopted.

## Model selection

- [ ] `model_picker.py` ranks by "chat-capable (name heuristic), then parameter count"; parameter count is only a
  proxy for quality and nothing here measures answer quality, so the default pick (BitNet-2B-4T on both machines) is
  a policy, not a benchmark result. A quality score per model (a downstream task, not perplexity, which is not
  comparable across tokenizers) would let it choose better, for example whether Llama3-8B beats 2B-4T for chat.
- [ ] The speed probe depends on the load at the time and, on multi-socket machines, on where the model's pages sit
  (32 against 53 t/s for 2B-4T on the Xeon); `--numa-evict` makes it representative. Cached readings can be stale
  for up to 7 days (`--reprobe`).
- [ ] Only models under `models/` with the canonical file names are considered; other GGUFs need `-m`.

## Untested

- [ ] ARM, Windows and macOS builds and runs. The ARM scalar path was only tested as an x86 build without AVX2,
  not on real ARM hardware (and not cross-compiled: no aarch64 sysroot or qemu on the dev machine).
- [ ] TL1/TL2 kernels, including the `setup_env.py -q tl2` codegen and compile path.
- [ ] MTEB or any downstream task. Only perplexity (WikiText-2, a 100 k-character slice) and embedding similarity
  checks were run; the MTEB table in `docs/bitnet-embeddings-i2s-guide.md` was not reproduced.
- [ ] Perplexity on the full WikiText-2 test set (1.29 M characters) rather than the slice.

## Housekeeping

- [ ] Delete stale model files: run `utils/cleanup_stale_models.sh` (dry run by default, `--yes` to delete; it only
  removes a file when the good model that replaced it is in place). It covers:
  - `bitnet_b1_58-3B/`: `ggml-model-f32.broken.gguf`, `ggml-model-i2_s.broken.gguf`,
    `ggml-model-i2_s.bad-ffn_down.gguf`, `ggml-model-i2_s.good-backup.gguf`, `ggml-model-i2_s.q8ffn-ref.gguf`
  - `Llama3-8B-1.58-100B-tokens/ggml-model-i2_s.bad-embd-i2s.gguf` (garbage: embedding quantized to I2_S)
  - `bitnet-b1.58-2B-4T-bf16/ggml-model-i2s-bitnet.gguf` (quantized before patch `0004`; garbage)
  - `bitnet_b1_58-large/ggml-model-i2_s.old.gguf`
  - `/tmp/claude-1000/3b_i2s_moved.gguf` (1.5 GB, in RAM-backed tmpfs)
- [ ] `setup_env.py` rewrites the tracked files `include/bitnet-lut-kernels.h` and `include/kernel_config.ini` on
  every run, so they show as modified after a build. Decide whether to stop tracking them.

## Known limits (not bugs)

- Llama3-8B needs about 36 GB resident for the f32 conversion, so it needs swap on a 64 GB machine.
- Perplexities are only comparable between files of the same model, because the tokenizers differ.
- `utils/test_power.sh` needs root or passwordless `sudo` for RAPL and turbostat; RAPL is Intel-only (turbostat on
  AMD is untested). Without access it falls back to a CPU-usage estimate and labels it as such.
