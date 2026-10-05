# TODO

What still needs doing in the `birman` fork, as of 2026-10-05. Everything else we found has been fixed and is
described in the notice at the top of `README.md`.

## Code gaps

- [ ] **The `*_interleaved` GEMM/GEMV functions have no callers.** They now assert on rows that are not a multiple
  of 128 instead of silently skipping the tail. Delete them or give them tail support if something starts using them.
- [ ] Row lengths for I2_S must be a multiple of 4 (asserted in `quantize_i2_s`).

## ARM / Apple Silicon

Done on an M1 Max (see `README.md`, Known issues): the build (patch `0009`, CPU-only on macOS), correctness against
the x86 reference, NEON `vec_dot`/`gemm` (patches `0010`/`0011`: 2B-4T pp512 392.6 / tg128 77.0 t/s, from 4.72 / 3.55 scalar)
and the macOS thread/memory detection in `start_llama.py`.

- [x] The other models on ARM: all six match the x86 perplexities, `-q8emb` files are equally accurate (table in
  `README.md`).
- [ ] Other ARM CPUs, and the NEON path without DOTPROD (`ggml_vdotq_s32` falls back to `vmull`; compiled, not run).
- [x] Token embedding as Q8_0 (`build.sh` writes `-q8emb`, the picker prefers it): 2B-4T generation 69.4 -> 86.6 t/s at
  equal perplexity. `build.sh -m DIR` adds the `-q8emb` file for any model (from `-f16emb` when present).
  Tied embeddings gain generation speed; untied ones (Falcon-E, Falcon3, Llama3-8B) only shrink.
- [ ] Generation: a tg profile (f16 embedding) was 43% barrier wait, 40% the f16 output matmul, 12% I2_S `vec_dot`.
  Untested: `--poll`, and 7 threads for generation (94.9 against 90.2 t/s at 8, not conclusive).
- [ ] More prompt speed. Done in `0011` (2x8 tiles, NEON activation quantization: pp512 232 -> 393 t/s). A profile
  of pp512 after it: `gemm` 64% (at about 150 of the 191 GMAC/s `sdot` peak per core), threads waiting at barriers
  16.5%, flash attention 5%, `quantize_row_i8_s` 4.6% before its NEON loop, RMS norm 3%. What is left is mostly
  barrier time and upstream ops, not the I2_S kernel. Interleaved means on an idle M1 Max: pp512 232.1 (`0010`), 381.5
  (2x8 tile, old quantizer), 392.6 t/s (`0011`); tg128 69.3 -> 77.0. Fewer threads did not help (8: 340, 7: 321, 6: 314); raising thread
  priority (`--prio`) needs root on macOS and is untested.
- [ ] TL1: `setup_env.py` runs `codegen_tl1.py` on arm64 but always passes `-DBITNET_ARM_TL1=OFF`, so TL1 is never
  compiled in; untested.
- [ ] Metal: the built `ggml-metal/` backend has no I2_S support (the I2_S shaders are only in the legacy, unbuilt
  `ggml-metal.m`/`.metal`), so macOS builds are CPU-only.
- [ ] macOS scripts: `utils/test_gemm_kernel.sh` looks for `libggml.so` and uses `-march=native`,
  `utils/test_power.sh` is Intel/Linux only, `utils/cleanup_stale_models.sh` uses GNU `stat`/`numfmt`, and
  `build.sh` needs a bash 4+ first on `PATH` (Homebrew's).

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
  on the i9 (209 against 205). A `--prompt-heavy` option, or choosing by workload, is untested. On systems other than
  Linux and macOS the thread count falls back to the logical CPU count.
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
