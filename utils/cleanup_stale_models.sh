#!/usr/bin/env bash
# Remove model files left over from debugging the I2_S conversion (broken, garbage or superseded GGUFs).
#
# Usage: utils/cleanup_stale_models.sh [-y|--yes] [-h|--help]
#   (default)   dry run: list what would be deleted and how much space it frees
#   -y, --yes   actually delete
#
# Every entry names one exact file and a "keep" file that must exist (and be big enough) first, so a file
# is only removed when the good model that replaced it is in place. Missing files are skipped, and only
# regular files are deleted (symlinks and directories are never touched).
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
M="$ROOT/models"
DELETE=0

case "${1:-}" in
    -y|--yes) DELETE=1 ;;
    -h|--help) sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    "") ;;
    *) echo "Unknown option: $1 (try --help)" >&2; exit 2 ;;
esac

# stale file | why | keep file that must exist | minimum size of the keep file in bytes
ENTRIES=(
  "$M/bitnet_b1_58-3B/ggml-model-f32.broken.gguf|707 KB failed f32 conversion|$M/bitnet_b1_58-3B/ggml-model-f32.gguf|10000000000"
  "$M/bitnet_b1_58-3B/ggml-model-i2_s.broken.gguf|707 KB failed quantization|$M/bitnet_b1_58-3B/ggml-model-i2_s.gguf|900000000"
  "$M/bitnet_b1_58-3B/ggml-model-i2_s.bad-ffn_down.gguf|garbage: pre-tail-support I2_S ffn_down|$M/bitnet_b1_58-3B/ggml-model-i2_s.gguf|900000000"
  "$M/bitnet_b1_58-3B/ggml-model-i2_s.good-backup.gguf|duplicate of the current ggml-model-i2_s.gguf|$M/bitnet_b1_58-3B/ggml-model-i2_s.gguf|900000000"
  "$M/bitnet_b1_58-3B/ggml-model-i2_s.q8ffn-ref.gguf|superseded: ffn_down kept at Q8_0|$M/bitnet_b1_58-3B/ggml-model-i2_s.gguf|900000000"
  "$M/Llama3-8B-1.58-100B-tokens/ggml-model-i2_s.bad-embd-i2s.gguf|garbage: token embedding quantized to I2_S|$M/Llama3-8B-1.58-100B-tokens/ggml-model-i2_s.gguf|3000000000"
  "$M/bitnet-b1.58-2B-4T-bf16/ggml-model-i2s-bitnet.gguf|garbage: quantized before patch 0004|$M/bitnet-b1.58-2B-4T-bf16/ggml-model-i2s-fixed.gguf|700000000"
  "$M/bitnet_b1_58-large/ggml-model-i2_s.old.gguf|superseded: Q6_K token embedding|$M/bitnet_b1_58-large/ggml-model-i2_s.gguf|200000000"
  "/tmp/claude-1000/3b_i2s_moved.gguf|temporary copy of the 3B model|$M/bitnet_b1_58-3B/ggml-model-i2_s.gguf|900000000"
)

size_of() { stat -c %s "$1" 2>/dev/null || echo 0; }
human() { numfmt --to=iec --suffix=B "$1" 2>/dev/null || echo "${1}B"; }

if [ "$DELETE" -eq 1 ]; then echo "Deleting stale model files"; else echo "Dry run (nothing is deleted; pass --yes to delete)"; fi
echo

total=0; removed=0; skipped=0; missing=0
for entry in "${ENTRIES[@]}"; do
    IFS='|' read -r file why keep keep_min <<< "$entry"
    rel="${file#"$ROOT"/}"
    if [ ! -e "$file" ] && [ ! -L "$file" ]; then
        printf '  absent   %s\n' "$rel"; missing=$((missing + 1)); continue
    fi
    if [ ! -f "$file" ] || [ -L "$file" ]; then
        printf '  SKIP     %s (not a regular file)\n' "$rel"; skipped=$((skipped + 1)); continue
    fi
    if [ ! -f "$keep" ] || [ "$(size_of "$keep")" -lt "$keep_min" ]; then
        printf '  SKIP     %s (replacement %s is missing or too small)\n' "$rel" "${keep#"$ROOT"/}"
        skipped=$((skipped + 1)); continue
    fi
    sz=$(size_of "$file")
    if [ "$DELETE" -eq 1 ]; then
        if rm -f -- "$file"; then
            printf '  deleted  %-70s %8s  %s\n' "$rel" "$(human "$sz")" "$why"
            total=$((total + sz)); removed=$((removed + 1))
        else
            printf '  FAILED   %s\n' "$rel"; skipped=$((skipped + 1))
        fi
    else
        printf '  would delete  %-66s %8s  %s\n' "$rel" "$(human "$sz")" "$why"
        total=$((total + sz)); removed=$((removed + 1))
    fi
done

echo
if [ "$DELETE" -eq 1 ]; then
    echo "Deleted $removed file(s), freed $(human "$total"). Skipped: $skipped, already absent: $missing."
else
    echo "$removed file(s) would be deleted, freeing $(human "$total"). Skipped: $skipped, already absent: $missing."
    [ "$removed" -gt 0 ] && echo "Run again with --yes to delete them."
fi
[ "$skipped" -eq 0 ]
