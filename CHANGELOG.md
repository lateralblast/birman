# Changelog

Changes in the `birman` fork relative to microsoft/BitNet. Versions are recorded in `VERSION` (`start_llama.py --version`);
tag a release with `git tag v$(cat VERSION)`. Dates are when the change was made.

## 0.4.0 - 2026-10-07

### Added
- `start_llama.py -m NAME` fetches and builds a model that is not under `models/` (new `model_fetch.py`: catalogue from
  `setup_env.py` plus the embedding models, size/memory/disk/speed estimates, asks `[y/N]` only when the model does not
  look suitable for the machine; `--yes`, `--no-fetch`, `--check`). The f32 intermediate is removed after a successful
  conversion. Tested with a real fetch of Falcon-E-1B-Base; not with the embedding models.

## 0.3.0 - 2026-10-07

### Added
- `start_llama.py --largest` (same as `--prefer largest`) and name lookup for `-m`: `-m falcon`, `-m "falcon 7b"`,
  `-m 2b-4t` find models under `models/` (all words must occur in the directory/file name; matches are ranked and probed
  like the automatic choice; also works with `--embedding`). `model_picker.matches()`.

## 0.2.0 - 2026-10-07

### Added
- `start_llama.py --embedding`: runs `llama-server --embedding` with an embedding model from `models/` (0.6B first),
  `-b/-ub 2048`, port 8081 by default; combines with `--open`/`--local`. `model_picker.find_embedding_models()`.
- `utils/rag_demo.py`: minimal retrieval-augmented generation against an embedding server and a chat server (vector cache
  in `~/.cache/start_llama/rag_vectors.json`). README section "Retrieval-augmented generation (RAG)" with measurements
  and the caveat that retrieval quality on dense markdown was mediocre.

### Measured
- Falcon3-7B-Instruct-1.58bit converted and run on the Xeon: tg128 37.0 t/s, pp512 153-163 t/s, WikiText-2 slice perplexity 10.97 (I2_S +0.7% against f32 on 8 chunks) (README).

## 0.1.0 - 2026-10-07

First versioned state of the fork.

### Added
- `start_llama.py`: machine-aware launcher (physical-core thread count, `--numa distribute` on multi-node Linux, `-ngl 0`,
  2B-4T chat template, warnings), macOS support, automatic model choice with `model_picker.py` (`--prefer`, `--min-tps`,
  `--no-probe`, `--reprobe`, `--models-dir`).
- `start_llama.py --local` / `--open`: listen on 127.0.0.1 or on all interfaces. `--open` generates an API key when none
  is given (stored in `~/.cache/start_llama/api_key`, mode 0600, printed, passed as `--api-key-file`), opens the port in
  ufw/firewalld if enabled (`--no-firewall` to skip), and removes the rule again when the server exits; `--close`
  removes rules left by an unclean exit.
- `utils/test_server_api.py`: `--api-key`, `--api-key-file`, `$LLAMA_API_KEY`, and a check that an unauthenticated
  completion is refused (19 checks with a key).
- README section "Listing the models on a machine": how to see which models a machine has and which one the picker would start.
- `numa_distribute.py` (`--numa distribute` by default on multi-socket machines, `--numa-evict`), `build.sh`
  (patches, venv, model download, Q8_0 token-embedding variant `-q8emb`), patches `0001`-`0011` for the llama.cpp
  submodule (I2_S tail handling, scalar fallback, Falcon-E pretokenizer, macOS build, NEON `vec_dot`/`gemm`),
  `utils/cleanup_stale_models.sh`, `utils/test_i2s_kernels.c`, `TODO.md`.

### Changed
- `setup_env.py` keeps the token embedding at f16 by default (I2_S embeddings gave garbage on untied models).
- `utils/e2e_benchmark.py`, `utils/test_gemm_kernel.sh`, `utils/test_power.sh` (turbostat/RAPL) fixed to run unmodified.
- README rebuilt with measured results for the i9-9900, a 2-socket Xeon E5-2682 v4 and an Apple M1 Max; the claim that
  `--numa distribute` speeds up prompt processing was corrected after re-measuring (1.0-1.05x, not 1.5-1.9x).

### Known gaps
See `TODO.md`. Firewall handling has not been run against a real enabled ufw or firewalld.
