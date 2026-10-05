![birman](birman.jpg)

> [!NOTE]
> **This is a fork of [microsoft/BitNet](https://github.com/microsoft/BitNet)** (`birman`), kept to build and run on current Python and NumPy 2.x. Everything below this notice is the upstream README, unchanged.
>
> **Why:** upstream's pinned `llama.cpp` submodule requires `numpy~=1.26.4`, which has no wheel for newer Pythons, so `pip install -r requirements.txt` fails to build numpy (seen on Python 3.14.6). While fixing that, `BitNet-b1.58-2B-4T` also turned out to produce looping/garbage output with the pinned submodule commit.
>
> **What changed**
> - `patches/llama.cpp/0002-requirements-numpy-2.patch` relaxes the submodule's numpy pin to `numpy>=2.0`.
> - `patches/llama.cpp/0003-llama-quantize-i2_s.patch` (upstream commit `cea12e83f`) registers `I2_S` in `llama-quantize`, and `0004-i2_s-quantize-block-layout.patch` makes the I2_S quantizer write the layout the kernels and `dequantize_row_i2_s` read (128-element blocks packed into 32 bytes). `0005-i2_s-quantize-row-length-assert.patch` makes `quantize_i2_s` abort on a row length that is not a multiple of 128 instead of quietly writing garbage, and `0006-i2s-row-tail-support.patch` then adds support for such rows: the last `n % 128` elements of each row are stored after the 128-element blocks (4 per byte, in order) and added by a scalar helper in `vec_dot`, `gemm` and llamafile's `tinyBLAS_I2S_AVX` (AVX2 only). Rows that are a multiple of 128 keep the old layout, so existing GGUFs still work. `0007-i2s-scalar-fallback.patch` fixes the non-AVX2 code (what runs on ARM, where the fork has no NEON I2_S kernel): the scalar `ggml_vec_dot_i2_i8_s` read the old sequential packing and returned `sum(w*y)` with `w` in {-1,0,1}, whereas the callers expect `sum(code*y)` with codes 0-2 and subtract the activation sum themselves, and the scalar `ggml_gemm_i2_i8_s` passed an activation count where `vec_dot` expects a weight-row count, so it was wrong in two independent ways and gave garbage on every model; both now follow the AVX2 layout and convention, tail included. `0008-falcon-e-pretokenizer.patch` maps the `falcon_e` pre-tokenizer name the converter writes to the fork's existing `FALCON_E` type (the name was missing, so Falcon-E would not load) and replaces that type's second regex with the one in Falcon-E's `tokenizer.json` (it used the GPT-2 default, which splits whitespace and newline runs differently). Before `0004` it packed sequentially, so any model quantized locally produced garbage. `utils/convert-ms-to-gguf-bitnet.py` now targets `MODEL_ARCH.BITNET_B158`; the old `BITNET_25` has no name in the fork's `gguf-py` and no C++ implementation.
> - `patches/llama.cpp/0001-bitnet-b158-squared-relu-ffn.patch` fixes the `bitnet-b1.58` graph, which reused the 3B model's SiLU FFN. `BitNet-b1.58-2B-4T` needs squared ReLU, which was the cause of its bad output. The patch applies it only to the 30-layer 2B-4T model (`LLM_TYPE_2B`); the 1bitLLM models share the arch but use SiLU and stay on it (an earlier, broader version of this patch broke `bitnet_b1_58-large`).
> - The submodule is a fork we can't push to, so fixes are kept as patches and applied idempotently by `setup_env.py` (`apply_patches()`).
> - `setup_env.py` keeps the token embedding at f16 by default (`--no-quant-embd` to disable). Without it, `llama-quantize` quantizes the token embedding of models with untied embeddings (`tie_word_embeddings: false`: Llama3-8B, Falcon3) to I2_S, although an embedding is not ternary: that produced garbage for `Llama3-8B-1.58-100B-tokens` and costs about 7% perplexity on Falcon3-1B. For models with tied embeddings (bitnet_b1_58-large/3B, 2B-4T) the embedding is also the output projection and the default was Q6_K, which is just as accurate as f16 (perplexity 11.8166 against 11.8181 on bitnet_b1_58-large), so there the new default only costs about 55 MB.
> - `build.sh` runs the whole sequence: submodule init, patches, venv and requirements, model download, `setup_env.py`.
> - `start_llama.py` starts any llama.cpp binary with flags chosen for the machine: `./start_llama.py -m MODEL.gguf` (the server), `--tool cli|completion|bench|perplexity` for the others, `--check` to print what it found and the command without running anything. It reads the CPU topology from `/sys` and uses the number of physical cores this process may use as `-t` (SMT siblings count once; an affinity mask or a cgroup CPU quota lowers it), which is where generation peaked on both machines tested (8 on the i9-9900; 32 on the 2-socket Xeon with `--numa distribute`; prompt-heavy work may gain a little from more, pass `-t`); adds `--numa distribute` on multi-node Linux machines (`numa_distribute.py`, `--no-numa`, `--numa-evict`, same environment variables); adds `-ngl 0` and, for the 2B-4T model on the server or cli, the chat template; and warns when the CPU has no AVX2 or is ARM (both run the scalar fallback, there is no NEON kernel) or when the model does not fit in free RAM. Everything it does not recognise goes to the llama binary unchanged, and a flag you give yourself (`-ngl`, `--numa`, `--chat-template-file`) is never overridden. On the Xeon it reports 2 sockets, 32 physical cores, 2 NUMA nodes and starts with `-t 32 --numa distribute`; restricted to one socket with `numactl --cpunodebind=0` it drops to 16 threads. Through `--tool bench` there, generation was 28.6 t/s with `--no-numa` and 56.9 t/s with the defaults after `--numa-evict` (prompt processing, 400 and 384 t/s, was within noise). On Linux only: elsewhere the topology cannot be read, so the thread count falls back to the logical CPU count.
> - `run_inference.py` uses `llama-completion` for plain prompts (current `llama-cli` is chat-only), maps `-p` to the system prompt in `-cnv` mode, and uses `chat-templates/bitnet-b1.58-2B-4T.jinja` for the 2B-4T model. The GGUF's embedded template ends the prompt with an EOS token, which made chat answers unrelated to the question.
> - `CLAUDE.md` documents the above for Claude Code.
>
> **Quick start:** `./build.sh`, then `python run_inference.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -p "You are a helpful assistant" -cnv`. To start the server (or any other llama.cpp binary) with the flags chosen for your machine, use `./start_llama.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf`; `--check` shows what it detected and the command it would run (`--tool cli|completion|bench|perplexity` picks another binary, see the `start_llama.py` entry above).
>
> **Test scripts (`utils/`)**
> - `test_gemm_kernel.sh` (`utils/test_gemm_kernel.sh -i 100 -o results.csv`) now finds the libraries itself: `$GGML_LIB_DIR` if set, else `build/bin`, else the older `build/3rdparty/llama.cpp/ggml/src`. It links `libggml-cpu` and `libggml-base` when present, since the I2_S kernels live in `libggml-cpu` in current llama.cpp (before, the link failed with `undefined reference to ggml_vec_dot_i2_i8_s`). The benchmark now calls the library's `ggml_gemm_i2_i8_s`; before, it timed its own copy of the loop, which is slower than the tiled kernel llama.cpp runs (about 190-207 GFLOPS against about 168 on an i9-9900, 8 threads). Throughput and latency per token now count activation rows (`nr`), not weight rows; the old single-token figure was about 29.6 million tokens/s. The `Std dev` and the `±` in the CSV are now measured over the iterations; they were `sqrt((max-min)^2/12)` in the binary and `(max-min)/4` in the CSV. It still checks no output, and every case uses `n` of 2048 or 8192, so the tail path from patch `0006` is not benchmarked (pass another `-n` to the binary to try it).
> - `test_gemm_kernel.sh` no longer passes `-fopenmp` (the benchmark is single-threaded and uses no OpenMP, and the flag broke the link with clang when `libomp` is not installed); `CXX=clang++` works on machines without `g++`.
> - `test_power.sh` (`utils/test_power.sh <model.gguf> <out.csv> "<pp threads>" "<tg threads>"`, from the repo root) measures power with Intel RAPL (`/sys/class/powercap`: package energy over the run divided by its duration, plus DRAM when the CPU has a dram zone), else turbostat (average `PkgWatt`/`RAMWatt`), else the old CPU usage x 200 W estimate, which is only a guess and is flagged as such. Both RAPL and turbostat normally need root; when not root the script uses `sudo -n` for those reads if passwordless sudo works. `POWER_SOURCE=auto|rapl|turbostat|estimate` forces a source and `POWER_NO_SUDO=1` disables sudo. The CSV gained trailing `PowerSource` and `DRAM(W)` columns; `Power(W)` and `Energy(J/t)` are package power only. On the i9-9900 RAPL and turbostat agreed within about 2 W (about 70 W package for `bitnet_b1_58-large`), while the old estimate gave 82-102 W. The measurement covers the whole `llama-bench` process, including model load, and everything else running on the machine, so use an idle system. RAPL is Intel-only; turbostat on AMD is untested.
> - `e2e_benchmark.py` used to exit with status 1 even when the benchmark succeeded: the `sys.exit(1)` in `run_command` was dedented out of its `except` block. It now exits 0 on success and 1 on failure.
> - `test_i2s_kernels.c` (new; build line in its header) checks `quantize_i2_s`, `dequantize_row_i2_s`, `vec_dot`, `gemv`, `gemm` and llamafile's sgemm against a scalar reference at row lengths from 64 to 8640, including row tails; run it after any change to the I2_S kernels, on an AVX2 build and on one built without AVX2.
> - `test_server_api.py` (new) is a smoke test for a running `llama-server`: health, models, completion and OpenAI-style completion, chat with a system message and multi-turn memory, streaming (SSE), tokenize/detokenize, four concurrent requests, and error handling. Results are in the "Server API smoke test" section below. The answer checks assume a chat-capable model such as 2B-4T.
> - `test_perplexity.py` is unchanged and works. It needs `data/<dataset>/test.txt` folders (`--data-dir`), which are not in the repo; the results below use the WikiText-2 test set from `Salesforce/wikitext` on Hugging Face. For `--test-embeddings`, `-m` must be an f32 GGUF: it re-quantizes it to I2_S once per embedding type and deletes the files it created.
>
> **Benchmarks (2026-10-05, CPU only, two machines)**
>
> **First machine, CPU:** Intel Core i9-9900 (Coffee Lake, 8 cores / 16 threads, 3.1 GHz base, 5.0 GHz max turbo, AVX2, no AVX-512 or VNNI), 256 KiB L1d, 2 MiB L2, 16 MiB L3. RAM: 4 x 16 GB DDR4-2666 (about 42.7 GB/s theoretical peak). Linux 7.0.0-34, clang 21.1.8, llama.cpp submodule `390c30775` (build 9918), `birman` at `28aff1c` plus the uncommitted `build.sh`/patch `0006` fixes. Turbo on, governor `powersave` (intel_pstate), no thread pinning, machine idle (98% idle, load about 1) and 35 GB of RAM free, although 28 GB of swap was in use from earlier work. No GPU (`-ngl 0`), 8 threads unless stated. One run per row: `llama-bench` mean +/- standard deviation over 3 repeats (5 for `e2e_benchmark.py`), so treat differences of a few percent as noise.
>
> **Speed, `llama-bench -p 512 -n 128 -t 8 -r 3`** (default batch size; I2_S weights, f16 token embedding unless noted):
>
> | Model | Params | File | Prompt pp512 (t/s) | Generation tg128 (t/s) |
> |---|---:|---:|---:|---:|
> | bitnet_b1_58-large | 0.73 B | 257 MiB | 393.9 +/- 6.0 | 83.6 +/- 1.3 |
> | Falcon3-1B-Instruct-1.58bit (1) | 1.67 B | 544 MiB | 307.8 +/- 14.4 | 53.7 +/- 0.1 |
> | BitNet-b1.58-2B-4T (official GGUF) | 2.41 B | 1.10 GiB | 188.7 +/- 6.6 | 23.4 +/- 0.1 |
> | bitnet_b1_58-3B | 3.32 B | 965 MiB | 93.7 +/- 1.4 | 25.9 +/- 0.0 |
> | Llama3-8B-1.58-100B-tokens | 8.03 B | 3.01 GiB | 61.1 +/- 1.2 | 12.3 +/- 0.1 |
>
> (1) That file was quantized before f16 embeddings became the default, so its embedding is I2_S (see the accuracy table: about 7% worse perplexity than with f16).
>
> On this single-node machine `start_llama.py` picks `-t 8` (the 8 physical cores) and adds no NUMA flags, so these tables are what it gives by default.
>
> **The repo's own `utils/e2e_benchmark.py -n 128 -p 128 -t 8`** forces a batch size of 1 (`-b 1`), so its prompt numbers are about as slow as generation and are not comparable to the table above: bitnet_b1_58-large 88.5 / 90.7 t/s (pp128 / tg128), Falcon3-1B 53.8 / 52.0, BitNet-2B-4T 23.1 / 22.5, bitnet_b1_58-3B 23.1 / 25.8, Llama3-8B 12.4 / 11.1.
>
> **Thread scaling, BitNet-2B-4T, `llama-bench -p 128 -n 128 -r 3`:**
>
> | Threads | 1 | 2 | 4 | 8 | 16 |
> |---|---:|---:|---:|---:|---:|
> | pp128 (t/s) | 37.2 | 76.9 | 136.3 | 204.7 | 208.9 |
> | tg128 (t/s) | 10.1 | 16.2 | 21.2 | 23.6 | 20.5 |
>
> Prompt processing scales almost linearly to the 8 physical cores and gains nothing from SMT (16 threads); generation stops scaling at about 4 threads and gets slower at 16. Llama3-8B generation moves roughly 3.23 GB of weights per token at 12.3 t/s, about 40 GB/s, which is close to the DDR4-2666 peak, so it looks memory-bandwidth-bound (I did not measure bandwidth separately).
>
> **GEMM kernel, `utils/test_gemm_kernel.sh -i 500`** (the library's `ggml_gemm_i2_i8_s`, n = 2048 unless noted): a single token takes 0.055 ms (153 GFLOPS); batches of 128 / 256 / 512 tokens take 7.2 / 11.1 / 22.6 ms (149 / 194 / 190 GFLOPS); the 8192-wide FFN cases take 21.2 ms (up, 203 GFLOPS) and 23.1 ms (down, 186 GFLOPS); 2048 tokens take 94.4 ms (182 GFLOPS); 32 tokens take 1.30 ms (207 GFLOPS). The 128-token case was 5.2-6.4 ms in earlier runs, so the small cases vary by 20% or so from run to run.
>
> **Power, `utils/test_power.sh`, BitNet-2B-4T, 8 threads, Intel RAPL:** prompt processing 202.6 t/s at 61.9 W package (3.3 W DRAM), 0.31 J/token; generation 23.1 t/s at 64.4 W package (6.1 W DRAM), 2.79 J/token. Package power only, whole `llama-bench` process including model load, nothing else running.
>
> **Second machine: 2 x Xeon E5-2682 v4 (2026-10-05)**
>
> **System:** 2 sockets x 16 cores x 2 threads = 64 logical CPUs (Broadwell, AVX2 and FMA, no AVX-512 or VNNI), 3.0 GHz max turbo, 40 MB L3 per socket, 2 NUMA nodes. RAM: 16 x 32 GB DDR4-2133 (about 68 GB/s theoretical per socket), 499 GB. Ubuntu, clang 21.1.8, same llama.cpp submodule (`390c30775`), `birman` at the commit that includes the `build.sh`, `test_gemm_kernel.sh` and NUMA changes described here. Idle machine (load 0.1), turbo on, governor `schedutil`. Same methods as above: `llama-bench` mean +/- standard deviation over 3 repeats, CPU only, I2_S weights. The models were copied from the i9-9900 machine (sizes and checksums verified) and the 2B-4T GGUF downloaded by `build.sh`.
>
> **Speed at 8 threads, `llama-bench -p 512 -n 128 -t 8 -r 3`** (same settings as the i9 table; Falcon-E-1B was not run on the i9):
>
> | Model | Xeon pp512 (t/s) | Xeon tg128 (t/s) | i9-9900 pp512 | i9-9900 tg128 |
> |---|---:|---:|---:|---:|
> | bitnet_b1_58-large | 330.8 +/- 4.0 | 85.0 +/- 2.9 | 393.9 | 83.6 |
> | Falcon3-1B (I2_S embedding) | 198.4 +/- 15.6 | 53.5 +/- 2.2 | 307.8 | 53.7 |
> | Falcon-E-1B-Instruct | 149.2 +/- 0.4 | 54.5 +/- 2.5 | - | - |
> | BitNet-b1.58-2B-4T | 131.3 +/- 0.5 | 23.4 +/- 0.2 | 188.7 | 23.4 |
> | bitnet_b1_58-3B | 66.2 +/- 0.2 | 23.7 +/- 0.9 | 93.7 | 25.9 |
> | Llama3-8B-1.58-100B-tokens | 49.1 +/- 0.0 | 12.9 +/- 0.3 | 61.1 | 12.3 |
>
> At 8 threads generation is within about 10% on both machines (it is limited by memory bandwidth), while prompt processing is 16-36% slower on the Xeon (lower clock and IPC). The repo's `e2e_benchmark.py -p 128 -n 128 -t 8` (batch size 1) gives 12.9 / 12.8 t/s for Llama3-8B and 22.8 / 23.4 for 2B-4T (pp128 / tg128).
>
> **Using the whole machine with the default placement (no NUMA options), `llama-bench -p 128 -n 128 -r 3`**, BitNet-2B-4T (pp128 / tg128, t/s):
>
> | Threads | 1 | 2 | 4 | 8 | 16 | 32 | 48 | 64 |
> |---|---:|---:|---:|---:|---:|---:|---:|---:|
> | unpinned, pp128 | 19.8 | 39.7 | 75.1 | 140.2 | 180.9 | 274.8 | 385.0 | 237.7 (+/- 88) |
> | unpinned, tg128 | 6.0 | 11.6 | 15.2 | 22.6 | 29.1 | 30.2 | 31.2 | 29.9 |
>
> and Llama3-8B:
>
> | Threads | 1 | 2 | 4 | 8 | 16 | 32 | 48 | 64 |
> |---|---:|---:|---:|---:|---:|---:|---:|---:|
> | unpinned, pp128 | 6.3 | 12.9 | 24.9 | 48.6 | 85.7 | 96.4 | 139.8 | 159.5 |
> | unpinned, tg128 | 2.8 | 5.6 | 8.1 | 13.2 | 17.3 | 17.8 | 18.6 | 17.2 |
>
> With the default placement (no NUMA options) prompt processing scales almost linearly up to 8 threads, then more slowly, and keeps gaining with SMT up to 48-64 threads; generation stops scaling at about 16 threads. That plateau is a placement problem, not a hardware limit:
>
> **NUMA placement is the biggest effect on this machine.** By default the model's pages sit on one NUMA node and the threads float between nodes, so generation is limited by one node's memory bandwidth (about 56 GB/s of weights for Llama3-8B; pinned to one socket it gets 41 GB/s, 12.8 t/s). llama.cpp's `--numa distribute` pins the threads evenly across the nodes, so each thread first-touches, and places in its own node's memory, the weight rows it always handles, and every read is local. `llama-bench -p 128 -n 128 -r 3`, the model's page cache evicted and then populated under each variant (t/s; ratio is `--numa distribute` over default):
>
> | Model | Threads | pp128 default | pp128 distribute | tg128 default | tg128 distribute | tg ratio |
> |---|---:|---:|---:|---:|---:|---:|
> | bitnet_b1_58-large | 32 | 749 | 953 | 112.8 | 143.6 | 1.27 |
> | bitnet_b1_58-large | 64 | 823 | 1289 | 81.2 | 141.7 | 1.74 |
> | BitNet-2B-4T | 32 | 286 | 416 | 30.4 | 58.4 | 1.92 |
> | BitNet-2B-4T | 64 | 458 | 516 | 28.5 | 51.2 | 1.79 |
> | bitnet_b1_58-3B | 32 | 183 | 299 | 35.9 | 55.9 | 1.56 |
> | bitnet_b1_58-3B | 64 | 266 | 308 | 32.7 | 54.8 | 1.68 |
> | Llama3-8B | 32 | 96.9 | 179.6 | 18.3 | 33.8 | 1.85 |
> | Llama3-8B | 64 | 144 | 172 | 15.9 | 31.7 | 1.99 |
>
> At 8 and 16 threads generation was 1.12-1.30x faster for all four models, and prompt processing was within 10% apart from 2B-4T and 3B at 16 threads (1.33x, 1.44x) and `bitnet_b1_58-large` at 16 threads, the one case where it was slower (466 against 660, 0.71x). Llama3-8B at 32 threads moves about 3.2 GB of weights per token at 33.8 t/s, 109 GB/s, against about 136 GB/s theoretical for the two sockets.
>
> **Where the pages were first touched matters.** The same test (32 threads, tg128 in t/s) with the model's pages first placed in different ways, and with `numactl --interleave=all` for comparison:
>
> | Starting state | Variant | 2B-4T | Llama3-8B |
> |---|---|---:|---:|
> | cache evicted, populated by the run | default | 29.4 | 17.3 |
> | | `--numa distribute` | **60.0** | **32.4** |
> | | `numactl --interleave=all` | 29.2 | 18.6 |
> | | both | 57.2 | 31.4 |
> | all pages preloaded onto node 0 (like right after copying the file) | default | 29.4 | 17.8 |
> | | `--numa distribute` | 33.6 | 19.4 |
> | | `numactl --interleave=all` | 29.4 | 18.5 |
> | | both | 34.0 | 19.5 |
>
> So `numactl --interleave=all` does nothing for generation, and `--numa distribute` roughly doubles it, but only when the pinned threads are the ones that place the pages: with the pages already on one node (just copied or downloaded, or placed by an earlier run without the flag) it gains about 10-15%. The page-cache state also explains why my first runs on this machine were so variable: unpinned numbers for the same model at 32 threads ranged from 275 to 467 pp128 depending on what had touched the file before (and my first suite wrongly showed interleaving helping prompt processing by 45-88%, which later runs with a settled cache did not reproduce). A fresh start needs no care; after copying or downloading a model, evict it once and let the next run place it.
>
> **The scripts now do this by default** on Linux machines with more than one NUMA node (`numa_distribute.py`): `run_inference.py`, `run_inference_server.py`, `start_llama.py`, `utils/e2e_benchmark.py` and `utils/test_power.sh` add `--numa distribute`. `--no-numa` (or `BITNET_NUMA=0`) turns it off, a `--numa` already given is respected, and `--numa-evict` (or `BITNET_NUMA_EVICT=1`) drops the model from the page cache before launching so the pinned threads place it. Through `run_inference.py -n 128 -t 32` on this machine (tokens per second): 2B-4T 28.6 with `--no-numa` on an evicted cache, 53.9 with the default and `--numa-evict`, 56.4 with the default again once the cache is placed; Llama3-8B 18.7, 33.7 and 33.8; `run_inference_server.py` starts with the flag and answers (44.5 t/s on 2B-4T). Tested only on this 2-node Xeon; machines with more nodes (for example AMD EPYC) or ARM servers are untested.
>
> **Speed with the new defaults on the Xeon** (`./start_llama.py --tool bench -m MODEL -p 512 -n 128 -r 3 --numa-evict`, which picked `-t 32 --numa distribute`; t/s; the i9-9900 columns are the table at the top, 8 threads):
>
> | Model | Xeon pp512 | Xeon tg128 | i9-9900 pp512 | i9-9900 tg128 |
> |---|---:|---:|---:|---:|
> | bitnet_b1_58-large | 945.5 +/- 140.9 | 135.5 +/- 0.3 | 393.9 | 83.6 |
> | Falcon3-1B (I2_S embedding) | 592.1 +/- 125.2 | 123.2 +/- 2.1 | 307.8 | 53.7 |
> | Falcon-E-1B-Instruct | 489.9 +/- 33.7 | 108.7 +/- 2.1 | - | - |
> | BitNet-b1.58-2B-4T | 382.0 +/- 62.2 | 60.3 +/- 0.6 | 188.7 | 23.4 |
> | bitnet_b1_58-3B | 238.3 +/- 26.1 | 57.6 +/- 2.0 | 93.7 | 25.9 |
> | Llama3-8B-1.58-100B-tokens | 168.0 +/- 11.9 | 33.3 +/- 0.1 | 61.1 | 12.3 |
>
> Against the 8-thread, default-placement Xeon table above, that is 2.9-3.6x the prompt throughput and 1.6-2.6x the generation speed, depending on the model; against the i9-9900 it is 1.9-2.8x on prompt processing and 1.6-2.7x on generation. The prompt-processing numbers vary by 7-21% between repeats (the `+/-`); generation is steady.
>
> **Does the Xeon build use everything the CPU has? Yes, and nothing else helped.** The build is `Release` (`-O3`) with `-march=native`, and llama.cpp reports `AVX2 = 1, FMA = 1, F16C = 1, BMI2 = 1` at startup (the I2_S kernels contain AVX2 `ymm` instructions); a Broadwell has no AVX-512 or VNNI, so there is nothing more at the instruction level. Alternatives were built or run beside it and benchmarked at 32 threads with `--numa distribute`, the cache warmed under each variant, three interleaved repeats for the build variants (pp512 / tg128 against the current build):
>
> | Variant | Result |
> |---|---|
> | LTO (`-DGGML_LTO=ON`) | mixed: prompt +5% (large), +11% (2B-4T), +3% (3B), -8% (Llama3-8B); generation -2% to +6% |
> | OpenMP through `libgomp` (clang has no `libomp` here, so the default build silently runs without OpenMP; `omp.h` copied from GCC) | about 10x slower: 2B-4T 18.8 / 6.2 against 470 / 58, Llama3-8B 6.1 / 2.9 |
> | larger micro-batch (`-ub 1024`, `-ub 2048 -b 2048`) | prompt -9%, generation unchanged |
> | flash attention (`-fa 1`) | no change |
> | no mmap (`-mmp 0`) | generation -9% (it loses the first-touch placement `--numa distribute` relies on) |
> | no mmap plus glibc transparent hugepages (`GLIBC_TUNABLES=glibc.malloc.hugetlb=1`) | generation -21% |
>
> So the default build and the default flags stay. Why the `libgomp` build is so slow was not investigated.
>
> **GEMM kernel, `utils/test_gemm_kernel.sh -i 500`** (single-threaded, n = 2048 unless noted): 60 GFLOPS for one token (0.14 ms), 81-87 GFLOPS for 128 to 2048 tokens, 101 GFLOPS for the 8192-wide `ffn_down` case, against 150-207 on the i9-9900 (5.0 GHz turbo).
>
> **Power, `utils/test_power.sh`, BitNet-2B-4T, Intel RAPL (both packages and both DRAM zones, via `sudo`):** at 8 threads prompt processing 133.4 t/s at 115.7 W (DRAM 47.6 W), 0.87 J/token, and generation 23.8 t/s at 118.9 W (DRAM 62.1 W), 4.99 J/token; at 32 threads 466.8 t/s at 195.0 W (0.42 J/token) and 31.9 t/s at 183.3 W (5.75 J/token). The sockets draw about 115 W with only 8 threads busy, so the extra threads are cheap for prompt processing and, with the default placement, a net loss for generation. With the new defaults (`-t 32 --numa distribute`, cache warmed under the pinned threads) the same test gives prompt processing 501.4 t/s at 197.0 W (DRAM 56.8 W), 0.39 J/token, and generation 60.2 t/s at 196.4 W (DRAM 82.3 W), 3.26 J/token: generation uses 43% less energy per token than the default placement (5.75 J), still above the i9-9900's 2.79 J/token (package power only; the Xeon figure covers two sockets).
>
> **Accuracy is identical on both machines.** All seven `test_perplexity.py` results on the WikiText-2 slice (large, Falcon3 with both embeddings, 2B-4T, 3B, Llama3-8B, Falcon-E-1B) and all five I2_S-against-f32 comparisons match the i9-9900 to four decimals, although the Xeon used 32 threads and a different CPU, so the I2_S kernels give the same results across hardware. The embedding-model similarity checks (270M and 0.6B) and `test_i2s_kernels.c` also pass on the Xeon.
>
> **Accuracy (2026-10-05)**
>
> **Perplexity, `utils/test_perplexity.py -d <data> -t 8 -c 512`** on the first 100 k characters of the WikiText-2 test set (about 25 k tokens, 48 chunks; +/- is the standard error). Perplexities are only comparable between files of the same model, because the tokenizers differ:
>
> | Model | Perplexity |
> |---|---:|
> | bitnet_b1_58-large | 11.82 +/- 0.27 |
> | bitnet_b1_58-3B | 8.84 +/- 0.19 |
> | BitNet-b1.58-2B-4T | 16.65 +/- 0.44 |
> | Llama3-8B-1.58-100B-tokens | 10.67 +/- 0.25 |
> | Falcon3-1B-Instruct-1.58bit, embedding f16 | 15.36 +/- 0.40 |
> | Falcon-E-1B-Instruct, embedding f16 | 9.82 +/- 0.21 |
> | Falcon3-1B-Instruct-1.58bit, embedding I2_S (old file) | 16.44 +/- 0.43 |
>
> **I2_S against the f32 GGUF it came from**, `llama-perplexity -c 512` on identical leading chunks (the f32 files are the converter's output, so this measures quantization loss only):
>
> | Model | Chunks | f32 | I2_S | Difference |
> |---|---:|---:|---:|---:|
> | bitnet_b1_58-large | 8 | 12.964 | 12.953 | -0.1% |
> | bitnet_b1_58-3B | 8 | 9.938 | 9.956 | +0.2% |
> | Falcon3-1B (f16 embedding) | 8 | 15.997 | 16.038 | +0.3% |
> | Falcon-E-1B (f16 embedding) | 8 | 11.000 | 11.044 | +0.4% |
> | Llama3-8B | 3 | 11.336 | 11.458 | +1.1% |
>
> All differences are well inside the standard errors (0.8-1.1 perplexity points on 3-8 chunks), so I2_S loses nothing measurable; the Llama3-8B wrong answer to "The capital of France is" is the model, not the quantization. With a tail-handling bug (before patch `0006`) bitnet_b1_58-3B scored about 7,400 on the same kind of test, so the check can see a broken kernel.
>
> **Token embedding type, `test_perplexity.py --test-embeddings`** (bitnet_b1_58-large, I2_S weights, WikiText-2 100 k): f32 11.818, f16 11.818, q8_0 11.808, q6_k 11.817, q5_0 11.838, q4_0 11.892, q3_k 12.033, **tq2_0 108.2**. Down to q6_k there is no measurable loss; below q4_0 it gets worse, and a ternary embedding is unusable.
>
> **Embedding models** (`llama-embedding`, `query: ` prefix, normalized): for both the 270M (640 dimensions) and the 0.6B (1024 dimensions) model, similar pairs (cat/kitten, Hund/dog, a password paraphrase) score 0.82-0.96 and unrelated pairs 0.62-0.75, so the order is right (270M: lowest similar 0.824 against highest unrelated 0.710; 0.6B: 0.886 against 0.749). That is a sanity check, not MTEB; the guide's MTEB table was not reproduced.
>
> **Server API smoke test (2026-10-05)**
>
> `llama-server` was started with `./start_llama.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf --port 8089` on both machines and exercised over HTTP with `utils/test_server_api.py` (18 checks, Python standard library only; `python utils/test_server_api.py [--url http://host:port]`, exit status 0 when all pass, 2 if the server cannot be reached). All 18 passed on both:
>
> | Check | i9-9900 | 2 x Xeon E5-2682 v4 |
> |---|---|---|
> | Flags `start_llama.py` chose | `-t 8`, no NUMA flags | `-t 32 --numa distribute` |
> | `GET /health`, `/v1/models`, `/props` | pass | pass |
> | `POST /completion`: "The capital of France is" answers Paris; `n_predict` respected; temperature 0 gives the same text twice | pass | pass |
> | `POST /v1/completions`: "The capital of Italy is" answers Rome | pass | pass |
> | `POST /v1/chat/completions` with a system message answers Paris and reports token usage | pass | pass |
> | Multi-turn chat remembers the name given earlier ("Your name is Alex.") | pass | pass |
> | Streaming chat (`"stream": true`): 24 server-sent-event chunks, then `[DONE]` | pass, first token after 0.15 s | pass, first token after 0.08 s |
> | `POST /tokenize` and `/detokenize` round-trip "Hello world" | pass | pass |
> | 4 concurrent `/completion` requests (continuous batching), all four answers correct | pass, 0.8 s | pass, 0.5 s |
> | Error handling: invalid JSON rejected, unknown route 404, malformed chat request 400, server healthy afterwards | pass | pass |
>
> The answers were word for word identical on both machines, and the 2B-4T chat template was applied automatically (chat answers such as "The capital of France is Paris." are clean). The first `/completion` request ran at 16 t/s on the i9 and 30 t/s on the Xeon; that is a short, cold 12-token request, so it understates steady-state generation (about 23 and 60 t/s in the benchmarks above). Upstream behaviour worth knowing: invalid JSON returns HTTP 500 with a parse error, where 400 would be more usual; the server stays healthy.
>
> **Verified:** Python 3.14.6, NumPy 2.5.3, clang 21, x86_64 Linux, I2_S kernel, with `microsoft/BitNet-b1.58-2B-4T-gguf`; `1bitLLM/bitnet_b1_58-large` through the full `python setup_env.py --hf-repo 1bitLLM/bitnet_b1_58-large -q i2_s` route (download, convert, quantize, run: correct output); and `microsoft/bitnet-embedding-0.6b` (embeddings sensible: cat/kitten 0.70, Hund/dog 0.77 across languages, unrelated pairs ~0.3). For 2B-4T: correct completion and multi-turn chat output at about 20 t/s on 8 threads, via `run_inference.py` and via `run_inference_server.py` (`/completion`, `/v1/chat/completions` including system message, multi-turn and streaming). `HF1BitLLM/Llama3-8B-1.58-100B-tokens` through `setup_env.py` (about 36 GB peak memory in the f32 conversion, so it needs swap on a 64 GB machine) with the embedding at f16: coherent, correct answers ("Water boils at a temperature of 100 degrees Celsius"; greedy decoding gave a wrong but fluent answer for "The capital of France is"). `1bitLLM/bitnet_b1_58-3B`: the f32 GGUF answers correctly; all-I2_S (966 MB) matches it ("Paris. It is the largest city in France"; perplexity 25.61 against 25.60 for a variant with `ffn_down` at Q8_0). Its `ffn_down` rows are 8640 long, which is not a multiple of 128; before patch `0006` the I2_S kernel dropped the last 64 elements of every row and the model was garbage (perplexity about 7400). Unit tests of `vec_dot`, `gemv`, `gemm`, llamafile sgemm and the dequantizer against a scalar reference (`utils/test_i2s_kernels.c`) pass for row lengths 64 to 8640, including tails that are not a multiple of 32, on both machines and on a build without AVX2. `./build.sh` was run from a fresh clone, twice, with exit status 0 each time, on the i9-9900 machine, and on the Xeon after two bugs in it were fixed (an interrupted `python -m venv` left a half-built `.venv` that it did not rebuild, and a `nullglob` side effect made its "is a model already there?" check always true, so it never downloaded the model). `tiiuae/Falcon-E-1B-Instruct` (untied embedding, so it needs the f16-embedding default) through `setup_env.py`: loads and answers correctly ("Paris"; "100 C (212 F) at sea level"); its tokenizer matches Hugging Face's `tokenizers` on 6 of 6 whole files (79,613 tokens: WikiText, README, C and Python source, multilingual text and symbols) and on 30 of 30 edge-case strings, and the converted model with an I2_S embedding was garbage, as Llama3-8B was. The non-AVX2 scalar path (`0007`): before the patch 29 unit checks failed and models produced nothing or `????????????`; after it all checks pass and bitnet_b1_58-large/3B and BitNet-2B-4T answer "Paris" on a build with AVX2 disabled.
>
> **Remaining issues**
> - Only the cases above were tested. Not tested: ARM, Windows/macOS, TL1/TL2 kernels, MTEB or any downstream task (only perplexity and embedding similarity checks were run, see Accuracy), and perplexity on the full WikiText-2 test set (a 100 k-character slice was used).
> - The GGUF's embedded chat template (`Human: … BITNETAssistant:`) is not the one the model was trained with, and its trailing EOS makes chat answers unrelated to the question. `chat-templates/bitnet-b1.58-2B-4T.jinja` is the template from the model's own `tokenizer_config.json` (`User: …<|eot_id|>Assistant: `), which `run_inference.py` and `run_inference_server.py` use for the 2B-4T model. With it the model ends turns cleanly, with no reverse prompt or `stop` strings. Other models still use their own embedded templates. In `run_inference_server.py`, `-p` is passed to `llama-server` as `-p` (upstream behaviour) and is not a system prompt; send system messages in the request.
> - HF checkpoint to I2_S conversion was tested via the README's safetensors route (`utils/convert-helper-bitnet.py` on `microsoft/bitnet-b1.58-2B-4T-bf16`: correct text, 95-99% of packed weight bytes match the official GGUF, the rest are ternary rounding differences) and via `setup_env.py --hf-repo` (`bitnet_b1_58-large`). The helper does not quantize the embedding to F16 as the official GGUF does (it stays Q6_K). The embedding models ship a prebuilt I2_S GGUF and convert in Python, so they don't use `llama-quantize`.
> - On ARM the I2_S mat-mul runs the scalar fallback (correct since patch `0007`, but slow); there is no NEON kernel in the build. The scalar path was tested by building for x86 without AVX2 (no `ymm` instructions in the kernels, unit tests and three models pass, perplexity within 0.1% of the AVX2 build), not on real ARM hardware. The `QK_I2_S = 64` define for NEON is only used by the AVX2 code, so it no longer matters. The `*_interleaved` GEMM/GEMV functions have no callers and now assert on rows that are not a multiple of 128. Row lengths must be a multiple of 4.
> - The I2_S kernels that actually run are in the submodule (`ggml-cpu/quants.c`, `ggml-cpu-i2s.c`, `llamafile/sgemm.cpp`). `src/ggml-bitnet-mad.cpp` is not compiled into the build (`src/CMakeLists.txt` overwrites it with the LUT source).
> - The NUMA default (`--numa distribute`) is verified only on one 2-socket, 2-node Xeon; machines with more nodes (for example AMD EPYC) and ARM servers are untested. It gains most when the pinned threads are the ones that first-touch the model's pages, so after copying or downloading a model run once with `--numa-evict` (about 10-15% instead of about 2x otherwise). `bitnet_b1_58-large` prompt processing at 16 threads was slower with it (0.71x); every other case measured was equal or faster.
> - `start_llama.py` chooses `-t` from the Linux CPU topology (physical cores); on other systems it falls back to the logical CPU count, and the thread count favours generation (prompt-heavy work gained a little from SMT threads on the Xeon).
> - On a fresh clone `pip install -r requirements.txt` still hits the old pin, because it runs before `setup_env.py` can patch it. Use `./build.sh`, or Python 3.10-3.12.
> - The `gguf` Python package from PyPI lacks the BitNet enums. `setup_env.py` (and so `build.sh`) installs the fork's `3rdparty/llama.cpp/gguf-py` into the venv, which fixes this; running the `utils/convert-*` scripts any other way needs `PYTHONPATH=3rdparty/llama.cpp/gguf-py`.
> - The `patches/` directory is a workaround; the real fix belongs in the llama.cpp fork the submodule points at.
> - `setup_env.py` rewrites the tracked file `include/bitnet-lut-kernels.h` on every run.
>
> **Build Requirements**
> - cmake
> - clang
> - python3.14-venv

<div align="center">

# bitnet.cpp

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![version](https://img.shields.io/badge/version-1.0-blue)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Collection-orange?logo=huggingface)](https://huggingface.co/collections/microsoft/bitnet)
[![Technical Report](https://img.shields.io/badge/Technical-Report-red?logo=arxiv)](https://arxiv.org/abs/2502.11880)
[![Demo](https://img.shields.io/badge/Online-Demo-green?logo=microsoft)](https://demo-bitnet-h0h8hcfqeqhrf5gf.canadacentral-01.azurewebsites.net/)
[![GPU Kernel](https://img.shields.io/badge/GPU-Kernel-6F42C1?logo=github)](https://github.com/microsoft/BitNet/blob/main/gpu/README.md)

</div>

<div align="left">

<h3>📰 News</h3>

<strong>07/23/2026:</strong> 📣 We released <a href="https://github.com/microsoft/VibeASR.cpp"><strong>VibeASR.cpp</strong></a> — a real-time multilingual ASR inference engine on CPU using BitNet I2_S quantization, achieving RTF < 1 with very few threads on x86 (AVX2) and ARM (NEON) platforms. [<a href="https://github.com/microsoft/VibeASR.cpp">Code</a>] [<a href="https://huggingface.co/microsoft/VibeVoice-ASR-BitNet">Models</a>] [<a href="https://arxiv.org/abs/2607.21075">Report</a>] ![NEW](https://img.shields.io/badge/NEW-red)

<strong>07/20/2026:</strong> 📣 We released <a href="https://huggingface.co/microsoft/BitNet-embedding-0.6B"><strong>BitNet-embedding-0.6B</strong></a> and <a href="https://huggingface.co/microsoft/BitNet-embedding-270M"><strong>BitNet-embedding-270M</strong></a> on Hugging Face — the first 1-bit embedding models that deliver competitive embedding quality with significantly faster inference on CPUs.
- **1.42x to 2.28x speedup** over F16 on BitNet-embedding-0.6B prefill (8 threads)
- **1.32x to 1.74x speedup** over F16 on BitNet-embedding-270M prefill (8 threads)
- Supports I2_S conversion with optimized kernels on x86 CPUs
- Lossless inference with 2 bits per weight

07/16/2026: 📣 Released [BitNet Embeddings 0.6B/270M: I2_S Conversion and Inference Optimization](docs/bitnet-embeddings-i2s-guide.md) — detailed guide for converting and running BitNet embedding models with optimized I2_S kernels.

01/15/2026: 📣 Released [BitNet CPU Inference Optimization](https://github.com/microsoft/BitNet/blob/main/src/README.md) — parallel kernel implementations with configurable tiling and embedding quantization support, achieving **1.15x to 2.1x** additional speedup over the original implementation.

05/20/2025: 📣 Released [BitNet Official GPU inference kernel](https://github.com/microsoft/BitNet/blob/main/gpu/README.md) — extending 1-bit inference beyond CPUs.

04/14/2025: 📣 Released [BitNet Official 2B Parameter Model](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T) on Hugging Face — the first official BitNet b1.58 model trained with 4T tokens.

02/18/2025: 📑 [Bitnet.cpp: Efficient Edge Inference for Ternary LLMs](https://arxiv.org/abs/2502.11880) — system-level paper on bitnet.cpp's architecture and design.

11/08/2024: 📑 [BitNet a4.8: 4-bit Activations for 1-bit LLMs](https://arxiv.org/abs/2411.04965) — enabling 4-bit activations for further efficiency gains.

10/21/2024: 📑 [1-bit AI Infra: Part 1.1, Fast and Lossless BitNet b1.58 Inference on CPUs](https://arxiv.org/abs/2410.16144) — the technical report behind bitnet.cpp.

10/17/2024: 📣 bitnet.cpp 1.0 released.

03/21/2024: 📑 [The-Era-of-1-bit-LLMs: Training Tips, Code, FAQ](https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf)

02/27/2024: 📑 [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764) — the foundational paper introducing BitNet b1.58.

10/17/2023: 📑 [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453) — the original BitNet paper.

</div>

## Overview

bitnet.cpp is the official inference framework for 1-bit LLMs (e.g., BitNet b1.58). It offers a suite of optimized kernels that support **fast** and **lossless** inference of 1.58-bit models on **CPU** and **GPU** (NPU support coming next).

Try it out via this [online demo](https://demo-bitnet-h0h8hcfqeqhrf5gf.canadacentral-01.azurewebsites.net/), or build and run it on your own [CPU](https://github.com/microsoft/BitNet?tab=readme-ov-file#build-from-source) or [GPU](https://github.com/microsoft/BitNet/blob/main/gpu/README.md).

bitnet.cpp achieves speedups of **1.37x** to **5.07x** on ARM CPUs, with larger models experiencing greater performance gains. Additionally, it reduces energy consumption by **55.4%** to **70.0%**, further boosting overall efficiency. On x86 CPUs, speedups range from **2.37x** to **6.17x** with energy reductions between **71.9%** to **82.2%**. Furthermore, bitnet.cpp can run a 100B BitNet b1.58 model on a single CPU, achieving speeds comparable to human reading (5-7 tokens per second), significantly enhancing the potential for running LLMs on local devices. Please refer to the [technical report](https://arxiv.org/abs/2410.16144) for more details.

<img src="./assets/performance.png" alt="performance_comparison" width="800"/>

## Model Releases

### 1. [BitNet-b1.58-2B-4T](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T) - 1-bit Large Language Model

**BitNet-b1.58-2B-4T** is the first official BitNet b1.58 model with **2.4B parameters**, trained on **4 trillion tokens**. It is a ternary (1.58-bit) language model that delivers competitive performance with full-precision models of similar size while enabling significantly faster and more energy-efficient inference.

- **Fast CPU Inference**: Achieves up to **6.17x speedup** on x86 CPUs and **5.07x** on ARM CPUs compared to full-precision models.
- **Energy Efficient**: Reduces energy consumption by up to **82.2%** on x86 and **70.0%** on ARM.
- **GPU Support**: Official GPU inference kernel available for accelerated deployment.
- **Chat-Ready**: Supports conversational mode for interactive use.

[🤗 Hugging Face](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T) | [🔗 Online Demo](https://demo-bitnet-h0h8hcfqeqhrf5gf.canadacentral-01.azurewebsites.net/) | [📄 Technical Report](https://arxiv.org/abs/2410.16144)

<img src="./assets/bitnet_b1.58_2b_benchmark.png" alt="BitNet b1.58 2B Benchmark" width="600"/>

### 2. [BitNet-embedding-0.6B](https://huggingface.co/microsoft/BitNet-embedding-0.6B) - 1-bit Embedding Model

**BitNet-embedding-0.6B** is a **0.6B-parameter** 1-bit embedding model that achieves competitive embedding quality with significantly faster CPU inference. It is the first model to demonstrate that ternary weights can deliver strong performance on embedding tasks.

- **1.42x to 2.28x speedup** over F16 on prefill (8 threads, x86)
- **Lossless Quality**: Competitive embedding quality with 2 bits per weight
- **I2_S Kernel**: Supports optimized I2_S conversion on x86 CPUs

[🤗 Hugging Face](https://huggingface.co/microsoft/BitNet-embedding-0.6B) | [📄 I2_S Guide](docs/bitnet-embeddings-i2s-guide.md)

<img src="./assets/embedding_prefill_0.6B.png" alt="BitNet Embedding 0.6B Prefill Performance" width="600"/>

### 3. [BitNet-embedding-270M](https://huggingface.co/microsoft/BitNet-embedding-270M) - Lightweight 1-bit Embedding Model

**BitNet-embedding-270M** is a compact **270M-parameter** 1-bit embedding model designed for resource-constrained environments, offering fast inference with minimal memory footprint.

- **1.32x to 1.74x speedup** over F16 on prefill (8 threads, x86)
- **Lossless Quality**: Competitive embedding quality with 2 bits per weight
- **Lightweight**: Only 270M parameters for edge deployment scenarios

[🤗 Hugging Face](https://huggingface.co/microsoft/BitNet-embedding-270M) | [📄 I2_S Guide](docs/bitnet-embeddings-i2s-guide.md)

<img src="./assets/embedding_prefill_270M.png" alt="BitNet Embedding 270M Prefill Performance" width="600"/>


## Supported Models

<table>
    <tr>
        <th rowspan="2">Model</th>
        <th rowspan="2">Parameters</th>
        <th rowspan="2">CPU</th>
        <th colspan="3">Kernel</th>
    </tr>
    <tr>
        <th>I2_S</th>
        <th>TL1</th>
        <th>TL2</th>
    </tr>
    <tr>
        <th colspan="6" style="text-align:left;">Official Models</th>
    </tr>
    <tr>
        <td rowspan="2"><a href="https://huggingface.co/microsoft/BitNet-b1.58-2B-4T">BitNet-b1.58-2B-4T</a></td>
        <td rowspan="2">2.4B</td>
        <td>x86</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
        <td>&#9989;</td>
    </tr>
    <tr>
        <td>ARM</td>
        <td>&#9989;</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
    </tr>
    <tr>
        <td rowspan="2"><a href="https://huggingface.co/microsoft/BitNet-embedding-0.6B">BitNet-embedding-0.6B</a></td>
        <td rowspan="2">0.6B</td>
        <td>x86</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
        <td>&#10060;</td>
    </tr>
    <tr>
        <td>ARM</td>
        <td>&#10060;</td>
        <td>&#10060;</td>
        <td>&#10060;</td>
    </tr>
    <tr>
        <td rowspan="2"><a href="https://huggingface.co/microsoft/BitNet-embedding-270M">BitNet-embedding-270M</a></td>
        <td rowspan="2">270M</td>
        <td>x86</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
        <td>&#10060;</td>
    </tr>
    <tr>
        <td>ARM</td>
        <td>&#10060;</td>
        <td>&#10060;</td>
        <td>&#10060;</td>
    </tr>
    <tr>
        <th colspan="6" style="text-align:left;">Community Models</th>
    </tr>
    <tr>
        <td rowspan="2"><a href="https://huggingface.co/1bitLLM/bitnet_b1_58-large">bitnet_b1_58-large</a></td>
        <td rowspan="2">0.7B</td>
        <td>x86</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
        <td>&#9989;</td>
    </tr>
    <tr>
        <td>ARM</td>
        <td>&#9989;</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
    </tr>
    <tr>
        <td rowspan="2"><a href="https://huggingface.co/1bitLLM/bitnet_b1_58-3B">bitnet_b1_58-3B</a></td>
        <td rowspan="2">3.3B</td>
        <td>x86</td>
        <td>&#10060;</td>
        <td>&#10060;</td>
        <td>&#9989;</td>
    </tr>
    <tr>
        <td>ARM</td>
        <td>&#10060;</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
    </tr>
    <tr>
        <td rowspan="2"><a href="https://huggingface.co/HF1BitLLM/Llama3-8B-1.58-100B-tokens">Llama3-8B-1.58-100B-tokens</a></td>
        <td rowspan="2">8.0B</td>
        <td>x86</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
        <td>&#9989;</td>
    </tr>
    <tr>
        <td>ARM</td>
        <td>&#9989;</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
    </tr>
    <tr>
        <td rowspan="2"><a href="https://huggingface.co/collections/tiiuae/falcon3-67605ae03578be86e4e87026">Falcon3 Family</a></td>
        <td rowspan="2">1B-10B</td>
        <td>x86</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
        <td>&#9989;</td>
    </tr>
    <tr>
        <td>ARM</td>
        <td>&#9989;</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
    </tr>
    <tr>
        <td rowspan="2"><a href="https://huggingface.co/collections/tiiuae/falcon-edge-series-6804fd13344d6d8a8fa71130">Falcon-E Family</a></td>
        <td rowspan="2">1B-3B</td>
        <td>x86</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
        <td>&#9989;</td>
    </tr>
    <tr>
        <td>ARM</td>
        <td>&#9989;</td>
        <td>&#9989;</td>
        <td>&#10060;</td>
    </tr>
</table>

❗️**We use existing 1-bit LLMs available on [Hugging Face](https://huggingface.co/) to demonstrate the inference capabilities of bitnet.cpp. We hope the release of bitnet.cpp will inspire the development of 1-bit LLMs in large-scale settings in terms of model size and training tokens.**

## Installation

### Requirements
- python>=3.10
- cmake>=3.22
- clang>=18
    - For Windows users, install [Visual Studio 2022](https://visualstudio.microsoft.com/downloads/). In the installer, toggle on at least the following options(this also automatically installs the required additional tools like CMake):
        -  Desktop-development with C++
        -  C++-CMake Tools for Windows
        -  Git for Windows
        -  C++-Clang Compiler for Windows
        -  MS-Build Support for LLVM-Toolset (clang)
    - For Debian/Ubuntu users, you can download with [Automatic installation script](https://apt.llvm.org/)

        `bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"`
- conda (highly recommend)

### Build from source

> [!IMPORTANT]
> If you are using Windows, please remember to always use a Developer Command Prompt / PowerShell for VS2022 for the following commands. Please refer to the FAQs below if you see any issues.

1. Clone the repo
```bash
git clone --recursive https://github.com/microsoft/BitNet.git
cd BitNet
```
2. Install the dependencies
```bash
# (Recommended) Create a new conda environment
conda create -n bitnet-cpp python=3.10
conda activate bitnet-cpp

pip install -r requirements.txt
```
3. Build the project
```bash
# Manually download the model and run with local path
huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir models/BitNet-b1.58-2B-4T
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s

```
<pre>
usage: setup_env.py [-h] [--hf-repo {1bitLLM/bitnet_b1_58-large,1bitLLM/bitnet_b1_58-3B,HF1BitLLM/Llama3-8B-1.58-100B-tokens,tiiuae/Falcon3-1B-Instruct-1.58bit,tiiuae/Falcon3-3B-Instruct-1.58bit,tiiuae/Falcon3-7B-Instruct-1.58bit,tiiuae/Falcon3-10B-Instruct-1.58bit}] [--model-dir MODEL_DIR] [--log-dir LOG_DIR] [--quant-type {i2_s,tl1}] [--quant-embd]
                    [--use-pretuned]

Setup the environment for running inference

optional arguments:
  -h, --help            show this help message and exit
  --hf-repo {1bitLLM/bitnet_b1_58-large,1bitLLM/bitnet_b1_58-3B,HF1BitLLM/Llama3-8B-1.58-100B-tokens,tiiuae/Falcon3-1B-Instruct-1.58bit,tiiuae/Falcon3-3B-Instruct-1.58bit,tiiuae/Falcon3-7B-Instruct-1.58bit,tiiuae/Falcon3-10B-Instruct-1.58bit}, -hr {1bitLLM/bitnet_b1_58-large,1bitLLM/bitnet_b1_58-3B,HF1BitLLM/Llama3-8B-1.58-100B-tokens,tiiuae/Falcon3-1B-Instruct-1.58bit,tiiuae/Falcon3-3B-Instruct-1.58bit,tiiuae/Falcon3-7B-Instruct-1.58bit,tiiuae/Falcon3-10B-Instruct-1.58bit}
                        Model used for inference
  --model-dir MODEL_DIR, -md MODEL_DIR
                        Directory to save/load the model
  --log-dir LOG_DIR, -ld LOG_DIR
                        Directory to save the logging info
  --quant-type {i2_s,tl1}, -q {i2_s,tl1}
                        Quantization type
  --quant-embd, --no-quant-embd
                        Keep the token embedding at f16 (default: on)
  --use-pretuned, -p    Use the pretuned kernel parameters
</pre>

## Usage
### Basic usage
```bash
# Run inference with the quantized model
python run_inference.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -p "You are a helpful assistant" -cnv
```
<pre>
usage: run_inference.py [-h] [-m MODEL] [-n N_PREDICT] -p PROMPT [-t THREADS] [-c CTX_SIZE] [-temp TEMPERATURE] [-cnv]

Run inference

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Path to model file
  -n N_PREDICT, --n-predict N_PREDICT
                        Number of tokens to predict when generating text
  -p PROMPT, --prompt PROMPT
                        Prompt to generate text from
  -t THREADS, --threads THREADS
                        Number of threads to use
  -c CTX_SIZE, --ctx-size CTX_SIZE
                        Size of the prompt context
  -temp TEMPERATURE, --temperature TEMPERATURE
                        Temperature, a hyperparameter that controls the randomness of the generated text
  -cnv, --conversation  Whether to enable chat mode or not (for instruct models.)
                        (When this option is turned on, the prompt specified by -p will be used as the system prompt.)
</pre>

### Demo

A demo of bitnet.cpp running a BitNet b1.58 3B model on Apple M2:

https://github.com/user-attachments/assets/7f46b736-edec-4828-b809-4be780a3e5b1

### Benchmark
We provide scripts to run the inference benchmark providing a model.

```  
usage: e2e_benchmark.py -m MODEL [-n N_TOKEN] [-p N_PROMPT] [-t THREADS]  
   
Setup the environment for running the inference  
   
required arguments:  
  -m MODEL, --model MODEL  
                        Path to the model file. 
   
optional arguments:  
  -h, --help  
                        Show this help message and exit. 
  -n N_TOKEN, --n-token N_TOKEN  
                        Number of generated tokens. 
  -p N_PROMPT, --n-prompt N_PROMPT  
                        Prompt to generate text from. 
  -t THREADS, --threads THREADS  
                        Number of threads to use. 
```  
   
Here's a brief explanation of each argument:  
   
- `-m`, `--model`: The path to the model file. This is a required argument that must be provided when running the script.  
- `-n`, `--n-token`: The number of tokens to generate during the inference. It is an optional argument with a default value of 128.  
- `-p`, `--n-prompt`: The number of prompt tokens to use for generating text. This is an optional argument with a default value of 512.  
- `-t`, `--threads`: The number of threads to use for running the inference. It is an optional argument with a default value of 2.  
- `-h`, `--help`: Show the help message and exit. Use this argument to display usage information.  
   
For example:  
   
```sh  
python utils/e2e_benchmark.py -m /path/to/model -n 200 -p 256 -t 4  
```  
   
This command would run the inference benchmark using the model located at `/path/to/model`, generating 200 tokens from a 256 token prompt, utilizing 4 threads.  

For the model layout that do not supported by any public model, we provide scripts to generate a dummy model with the given model layout, and run the benchmark on your machine:

```bash
python utils/generate-dummy-bitnet-model.py models/bitnet_b1_58-large --outfile models/dummy-bitnet-125m.tl1.gguf --outtype tl1 --model-size 125M

# Run benchmark with the generated model, use -m to specify the model path, -p to specify the prompt processed, -n to specify the number of token to generate
python utils/e2e_benchmark.py -m models/dummy-bitnet-125m.tl1.gguf -p 512 -n 128
```

### Convert from `.safetensors` Checkpoints

```sh
# Prepare the .safetensors model file
huggingface-cli download microsoft/bitnet-b1.58-2B-4T-bf16 --local-dir ./models/bitnet-b1.58-2B-4T-bf16

# Convert to gguf model
python ./utils/convert-helper-bitnet.py ./models/bitnet-b1.58-2B-4T-bf16
```

## Acknowledgements

This project is based on the [llama.cpp](https://github.com/ggerganov/llama.cpp) framework. We would like to thank all the authors for their contributions to the open-source community. Also, bitnet.cpp's kernels are built on top of the Lookup Table methodologies pioneered in [T-MAC](https://github.com/microsoft/T-MAC/). For inference of general low-bit LLMs beyond ternary models, we recommend using T-MAC.

### FAQ (Frequently Asked Questions)📌 

#### Q1: The build dies with errors building llama.cpp due to issues with std::chrono in log.cpp?

**A:**
This is an issue introduced in recent version of llama.cpp. Please refer to this [commit](https://github.com/tinglou/llama.cpp/commit/4e3db1e3d78cc1bcd22bcb3af54bd2a4628dd323) in the [discussion](https://github.com/abetlen/llama-cpp-python/issues/1942) to fix this issue.

#### Q2: How to build with clang in conda environment on windows?

**A:** 
Before building the project, verify your clang installation and access to Visual Studio tools by running:
```
clang -v
```

This command checks that you are using the correct version of clang and that the Visual Studio tools are available. If you see an error message such as:
```
'clang' is not recognized as an internal or external command, operable program or batch file.
```

It indicates that your command line window is not properly initialized for Visual Studio tools.

• If you are using Command Prompt, run:
```
"C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat" -startdir=none -arch=x64 -host_arch=x64
```

• If you are using Windows PowerShell, run the following commands:
```
Import-Module "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\Microsoft.VisualStudio.DevShell.dll" Enter-VsDevShell 3f0e31ad -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64"
```

These steps will initialize your environment and allow you to use the correct Visual Studio tools.
