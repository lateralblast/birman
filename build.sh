#!/usr/bin/env bash
# One-shot build: submodule -> patches -> venv + requirements -> model -> setup_env.py -> Q8_0-embedding copy.
# Safe to re-run; every step is skipped when already done.
#
# Usage: ./build.sh [-m MODEL_DIR] [-r HF_GGUF_REPO] [-q i2_s|tl1|tl2] [-v VENV_DIR] [-s] [-h]
#   -m  model directory                    (default: models/BitNet-b1.58-2B-4T)
#   -r  Hugging Face repo with a prebuilt GGUF, downloaded if the dir has none
#                                          (default: microsoft/BitNet-b1.58-2B-4T-gguf)
#   -q  quantization type passed to setup_env.py (default: i2_s)
#   -v  virtualenv location                (default: .venv)
#   -s  skip the model download
set -euo pipefail

MODEL_DIR="models/BitNet-b1.58-2B-4T"
HF_REPO="microsoft/BitNet-b1.58-2B-4T-gguf"
QUANT="i2_s"
VENV=".venv"
SKIP_DOWNLOAD=0

while getopts "m:r:q:v:sh" opt; do
  case $opt in
    m) MODEL_DIR=$OPTARG ;;
    r) HF_REPO=$OPTARG ;;
    q) QUANT=$OPTARG ;;
    v) VENV=$OPTARG ;;
    s) SKIP_DOWNLOAD=1 ;;
    h) sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) exit 2 ;;
  esac
done

cd "$(dirname "$0")"
ROOT=$PWD
SUB=3rdparty/llama.cpp
log() { printf '\n==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

# 1. tools -------------------------------------------------------------------
log "Checking tools"
for t in git cmake clang clang++; do
  command -v "$t" >/dev/null || die "$t not found (clang>=18 and cmake>=3.22 are required; setup_env.py hard-codes clang)"
done

# Prefer 3.10-3.12: the submodule's original numpy pin has no wheel on newer Pythons.
PY=""
for c in python3.10 python3.11 python3.12 python3; do
  command -v "$c" >/dev/null && { PY=$c; break; }
done
[ -n "$PY" ] || die "python3 not found"
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || die "python>=3.10 required"
echo "python: $($PY --version)  clang: $(clang --version | head -1)"

# 2. submodule ---------------------------------------------------------------
log "Initialising submodule"
git submodule update --init --recursive

# 3. patches (must precede pip: patch 0002 relaxes the numpy pin) -------------
log "Applying patches/llama.cpp"
shopt -s nullglob
# A later patch may edit lines an earlier one added, which stops the earlier one from
# reverse-applying, so a patch also counts as applied when any later patch is applied.
patches=("$ROOT"/patches/llama.cpp/*.patch)
for i in "${!patches[@]}"; do
  p=${patches[$i]}
  name=$(basename "$p")
  # check "superseded" before the forward apply: a patch whose lines a later patch rewrote can still pass
  # `git apply --check` and would be applied a second time (0010 was, on top of 0011)
  superseded=0
  for q in "${patches[@]:$((i + 1))}"; do
    if git -C "$SUB" apply --reverse --check "$q" 2>/dev/null; then superseded=1; break; fi
  done
  if git -C "$SUB" apply --reverse --check "$p" 2>/dev/null; then
    echo "already applied: $name"
  elif [ "$superseded" -eq 1 ]; then
    echo "already applied (superseded by a later patch): $name"
  elif git -C "$SUB" apply --check "$p" 2>/dev/null; then
    git -C "$SUB" apply "$p" && echo "applied: $name"
  else
    die "$name does not apply to the submodule at $(git -C "$SUB" rev-parse --short HEAD)"
  fi
done

# 4. python environment ------------------------------------------------------
log "Installing requirements into $VENV"
# venv-begin
# An interrupted `python -m venv` leaves a directory without bin/activate, so test for that file, not the
# directory. Only a directory that is clearly a half-built venv (it has pyvenv.cfg) is removed and rebuilt.
if [ ! -f "$VENV/bin/activate" ]; then
  if [ -f "$VENV/pyvenv.cfg" ]; then
    echo "removing incomplete virtualenv $VENV"
    rm -rf "$VENV"
  elif [ -d "$VENV" ] && [ -n "$(ls -A "$VENV" 2>/dev/null)" ]; then
    die "$VENV exists and is not a virtualenv; remove it or pass another location with -v"
  fi
  "$PY" -m venv "$VENV" || die "could not create a virtualenv in $VENV (on Debian/Ubuntu: apt install python3-venv)"
fi
# venv-end
# shellcheck disable=SC1091
. "$VENV/bin/activate"
pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q "huggingface_hub[cli]" >/dev/null 2>&1 || true

# 5. model -------------------------------------------------------------------
# compgen -G, not ls: with nullglob (set above) an unmatched `ls dir/*.gguf` becomes a bare `ls` and succeeds
if compgen -G "$MODEL_DIR/ggml-model-*.gguf" >/dev/null; then
  log "Model already present in $MODEL_DIR"
elif [ "$SKIP_DOWNLOAD" -eq 1 ]; then
  log "Skipping model download (-s)"
else
  log "Downloading $HF_REPO -> $MODEL_DIR"
  python - "$HF_REPO" "$MODEL_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2])
PY
fi

# 6. configure + build -------------------------------------------------------
log "Running setup_env.py (kernel codegen, cmake build)"
python setup_env.py -md "$MODEL_DIR" -q "$QUANT"

# 7. Q8_0 token embedding ----------------------------------------------------
# An f16 token embedding that is also the output projection (2B-4T: 657 MB of 1188) is read for every generated
# token. As Q8_0 it gave the same perplexity (16.6436 against 16.6467) and 25% faster generation on an M1 Max.
# llama-quantize copies the I2_S tensors unchanged (same target type) and converts only the embedding. Only f16/f32
# embeddings are converted: an I2_S embedding cannot be dequantized (its to_float takes a scale argument).
SRC="$MODEL_DIR/ggml-model-i2_s-f16emb.gguf"   # Falcon3: the plain file has an I2_S embedding
[ -f "$SRC" ] || SRC="$MODEL_DIR/ggml-model-i2_s.gguf"
DST="$MODEL_DIR/ggml-model-i2_s-q8emb.gguf"
if [ "$QUANT" = "i2_s" ] && [ -f "$SRC" ] && [ ! -f "$DST" ]; then
  ETYPE=$(python - "$SRC" <<'PY'
import sys
from gguf import GGUFReader
print(next((t.tensor_type.name for t in GGUFReader(sys.argv[1]).tensors if t.name == "token_embd.weight"), "none"))
PY
)
  if [ "$ETYPE" = "F16" ] || [ "$ETYPE" = "F32" ]; then
    log "Writing $DST (token embedding $ETYPE -> Q8_0)"
    build/bin/llama-quantize --allow-requantize --token-embedding-type q8_0 "$SRC" "$DST.tmp" I2_S > logs/quantize_q8emb.log 2>&1 \
      || die "llama-quantize failed, see logs/quantize_q8emb.log"
    mv "$DST.tmp" "$DST"
  else
    log "Not writing $DST: token embedding is $ETYPE"
  fi
fi

# 8. report ------------------------------------------------------------------
GGUF=$(compgen -G "$MODEL_DIR/ggml-model-i2_s-q8emb.gguf" || compgen -G "$MODEL_DIR/ggml-model-*.gguf" | head -1 || true)
log "Done"
echo "binaries: $ROOT/build/bin/{llama-cli,llama-completion,llama-server,llama-quantize}"
if [ -n "$GGUF" ]; then
  echo "try:      build/bin/llama-completion -m $GGUF -n 32 -p \"The capital of France is\" -t 8 -ngl 0 --temp 0 -no-cnv"
fi
git diff --quiet -- include/bitnet-lut-kernels.h || \
  echo "note:     setup_env.py regenerated include/bitnet-lut-kernels.h (tracked); 'git checkout include/bitnet-lut-kernels.h' to discard"
