#!/bin/bash
# Monitor power consumption for llama-bench with different thread configurations
# Usage: ./test_power.sh <model_path> <output_csv> <pp_threads> <tg_threads>
# Example: ./test_power.sh models/model.gguf results.csv "1,2,4,8" "1,2,4,8"
#
# Power is measured, in order of preference, with:
#   rapl       Intel RAPL energy counters (/sys/class/powercap): package energy over the run
#              divided by its duration, plus DRAM when the CPU exposes a dram zone
#   turbostat  average PkgWatt / RAMWatt that turbostat reports during the run
#   estimate   CPU usage x 200 W, a rough guess, used only when neither of the above works
# Reading RAPL and running turbostat normally needs root. When not root, passwordless sudo
# (sudo -n) is used for those reads if available.
# Environment:
#   POWER_SOURCE=auto|rapl|turbostat|estimate   force a source (default: auto)
#   POWER_NO_SUDO=1                             never use sudo
# Power(W) and Energy(J/t) are CPU package power only (DRAM is reported separately, and is not
# part of Energy). The package counters include everything else running on the machine, so
# run on an otherwise idle system. The PowerSource column says how each row was measured.
set -e

# Parse arguments
if [ $# -ne 4 ]; then
    echo "Usage: $0 <model_path> <output_csv> <pp_threads> <tg_threads>"
    echo "Example: $0 models/model.gguf results.csv \"1,2,4,8\" \"1,2,4,8\""
    exit 1
fi

MODEL_PATH="$1"
OUTPUT_CSV="$2"
PP_THREADS="$3"
TG_THREADS="$4"

TEMP_LOG="/tmp/power_monitor_$$.log"
PID_FILE="/tmp/monitor_$$.pid"
BENCH_OUTPUT="/tmp/bench_output_$$.txt"

# Validate model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found: $MODEL_PATH"
    exit 1
fi

# Create output directory if needed
mkdir -p "$(dirname "$OUTPUT_CSV")"

TMPD=$(mktemp -d)
TS_LOG="$TMPD/turbostat.log"
TS_PIDFILE="$TMPD/turbostat.pid"
TS_BG_PID=""
RAPL_ROOT=/sys/class/powercap
POWER_SOURCE="${POWER_SOURCE:-auto}"
SUDO=""
RAPL_PKG_ZONES=()
RAPL_DRAM_ZONES=()

stop_turbostat() {
    if [ -n "$TS_BG_PID" ]; then
        [ -s "$TS_PIDFILE" ] && as_root kill -TERM "$(cat "$TS_PIDFILE")" 2>/dev/null || true
        wait "$TS_BG_PID" 2>/dev/null || true
        TS_BG_PID=""
    fi
}
cleanup() { stop_turbostat; rm -rf "$TMPD"; }
trap cleanup EXIT

# Run a command as root when sudo is in use, otherwise as the current user
as_root() { if [ -n "$SUDO" ]; then $SUDO "$@"; else "$@"; fi; }

sudo_ok() { [ -z "${POWER_NO_SUDO:-}" ] && command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; }

find_rapl_zones() {
    local z name
    for z in "$RAPL_ROOT"/intel-rapl:*; do
        [ -e "$z/energy_uj" ] || continue
        name=$(cat "$z/name" 2>/dev/null)
        case "$name" in
            package-*) RAPL_PKG_ZONES+=("$z") ;;
            dram)      RAPL_DRAM_ZONES+=("$z") ;;
        esac
    done
}

# RAPL is usable if there is a package zone whose counter we can read (directly or via sudo)
rapl_usable() {
    find_rapl_zones
    [ ${#RAPL_PKG_ZONES[@]} -gt 0 ] || return 1
    local f="${RAPL_PKG_ZONES[0]}/energy_uj"
    if cat "$f" >/dev/null 2>&1; then return 0; fi
    if sudo_ok && sudo -n cat "$f" >/dev/null 2>&1; then SUDO="sudo -n"; return 0; fi
    return 1
}

# Print "<energy_uj> <max_energy_range_uj>" for every zone given
rapl_read() {
    local z
    for z in "$@"; do
        echo "$(as_root cat "$z/energy_uj") $(cat "$z/max_energy_range_uj" 2>/dev/null || echo 0)"
    done
}

# Joules consumed between two rapl_read outputs, summed over zones, handling counter wrap
rapl_joules() {
    paste -d' ' <(echo "$1") <(echo "$2") | awk '{
        d = $3 - $1; if (d < 0) d += $2; sum += d
    } END { printf "%.6f", sum / 1e6 }'
}

# turbostat is usable if it prints a numeric PkgWatt row (directly or via sudo)
turbostat_works() {
    "$@" turbostat --quiet --Summary --show PkgWatt --interval 0.2 --num_iterations 1 2>/dev/null |
        awk 'NR == 2 && $1 ~ /^[0-9.]+$/ { ok = 1 } END { exit !ok }'
}
turbostat_usable() {
    command -v turbostat >/dev/null 2>&1 || return 1
    if turbostat_works env; then return 0; fi
    if sudo_ok && turbostat_works sudo -n; then SUDO="sudo -n"; return 0; fi
    return 1
}

start_turbostat() {
    : > "$TS_LOG"; rm -f "$TS_PIDFILE"
    # the wrapper records turbostat's pid (it may run as root) so it can be stopped later
    as_root sh -c 'echo $$ > "$1"; shift; exec turbostat "$@"' sh "$TS_PIDFILE" \
        --quiet --Summary --show PkgWatt,RAMWatt --interval 0.5 > "$TS_LOG" 2>/dev/null &
    TS_BG_PID=$!
    local i
    for i in 1 2 3 4 5 6 7 8 9 10; do [ -s "$TS_PIDFILE" ] && break; sleep 0.2; done
}

# Average "<PkgWatt> <RAMWatt>" over the complete rows in the turbostat log (columns by name)
turbostat_avg() {
    awk -F'\t' 'NR == 1 { for (i = 1; i <= NF; i++) { if ($i == "PkgWatt") p = i; if ($i == "RAMWatt") r = i } ncol = NF; next }
        NF == ncol && p && $p ~ /^[0-9.]+$/ { sp += $p; n++; if (r && $r ~ /^[0-9.]+$/) { sr += $r; nr++ } }
        END { if (n) printf "%.2f %s", sp / n, (nr ? sprintf("%.2f", sr / nr) : "NA"); else printf "0 NA" }' "$TS_LOG"
}

# CPU-usage monitor, only used for the rough estimate
monitor_cpu() {
    local log_file="$1"
    echo "Timestamp,CPU_Usage(%),Avg_Freq(MHz)" > "$log_file"
    while [ -f "$PID_FILE" ]; do
        cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print 100-$8}')
        avg_freq=$(grep "cpu MHz" /proc/cpuinfo | awk '{sum+=$4; count++} END {printf "%.0f", sum/count}')
        timestamp=$(date +%s.%N)
        echo "$timestamp,$cpu_usage,$avg_freq" >> "$log_file"
        sleep 0.5
    done
}

# Function to calculate average power from the CPU-usage log (a guess: usage x 200 W)
calculate_power() {
    local log_file="$1"
    awk -F',' 'NR>1 {sum_cpu+=$2; count++} END {
        if (count > 0) {
            avg_cpu = sum_cpu/count
            est_power = avg_cpu * 200 / 100
            printf "%.2f", est_power
        } else {
            print "0"
        }
    }' "$log_file"
}

# Pick the power source
POWER_USED=""
case "$POWER_SOURCE" in
    auto)
        if rapl_usable; then POWER_USED=rapl
        elif turbostat_usable; then POWER_USED=turbostat
        else POWER_USED=estimate; fi ;;
    rapl)
        rapl_usable || { echo "Error: RAPL counters are not readable (try root or passwordless sudo)" >&2; exit 1; }
        POWER_USED=rapl ;;
    turbostat)
        turbostat_usable || { echo "Error: turbostat is not usable (needs root or passwordless sudo)" >&2; exit 1; }
        POWER_USED=turbostat ;;
    estimate) POWER_USED=estimate ;;
    *) echo "Error: POWER_SOURCE must be auto, rapl, turbostat or estimate" >&2; exit 1 ;;
esac
case "$POWER_USED" in
    rapl)      echo "Power source: Intel RAPL (${#RAPL_PKG_ZONES[@]} package, ${#RAPL_DRAM_ZONES[@]} dram zone(s))${SUDO:+, read with $SUDO}" >&2 ;;
    turbostat) echo "Power source: turbostat${SUDO:+, run with $SUDO}" >&2 ;;
    estimate)  echo "Warning: no RAPL or turbostat access; power is a rough estimate (CPU usage x 200 W), not a measurement" >&2 ;;
esac

# Function to extract throughput from llama-bench output
extract_throughput() {
    local bench_output="$1"
    local workload="$2"
    grep "$workload" "$bench_output" | awk '{
        # Extract mean from "mean ± std" format
        for (i=1; i<=NF; i++) {
            if ($(i+1) == "±") {
                printf "%.2f", $i
                exit
            }
        }
    }'
}

# Function to run single benchmark
run_benchmark() {
    local workload="$1"  # "pp" or "tg"
    local threads="$2"
    local n_flag=""
    
    if [ "$workload" = "pp" ]; then
        n_flag="-n 0"
        workload_name="pp128"
    else
        n_flag="-n 128"
        workload_name="tg128"
    fi
    
    # Output progress to stderr (won't be captured in CSV)
    echo "Testing $workload_name with $threads threads..." >&2
    
    # Start measuring
    local monitor_pid="" rapl_pkg0="" rapl_dram0="" t0=""
    case "$POWER_USED" in
        rapl)
            rapl_pkg0=$(rapl_read "${RAPL_PKG_ZONES[@]}")
            [ ${#RAPL_DRAM_ZONES[@]} -gt 0 ] && rapl_dram0=$(rapl_read "${RAPL_DRAM_ZONES[@]}")
            t0=$(date +%s.%N) ;;
        turbostat) start_turbostat ;;
        estimate)
            touch "$PID_FILE"
            monitor_cpu "$TEMP_LOG" &
            monitor_pid=$! ;;
    esac
    
    # Run benchmark
    ./build/bin/llama-bench -m "$MODEL_PATH" -p 128 $n_flag -t "$threads" -ngl 0 > "$BENCH_OUTPUT" 2>&1
    
    # Stop measuring and work out average power (W): package, and DRAM where available
    local power="0" dram="NA"
    case "$POWER_USED" in
        rapl)
            local t1=$(date +%s.%N)
            local rapl_pkg1=$(rapl_read "${RAPL_PKG_ZONES[@]}")
            local secs=$(awk -v a="$t0" -v b="$t1" 'BEGIN {printf "%.6f", b - a}')
            power=$(awk -v j="$(rapl_joules "$rapl_pkg0" "$rapl_pkg1")" -v s="$secs" 'BEGIN {if (s > 0) printf "%.2f", j / s; else print "0"}')
            if [ ${#RAPL_DRAM_ZONES[@]} -gt 0 ]; then
                local rapl_dram1=$(rapl_read "${RAPL_DRAM_ZONES[@]}")
                dram=$(awk -v j="$(rapl_joules "$rapl_dram0" "$rapl_dram1")" -v s="$secs" 'BEGIN {if (s > 0) printf "%.2f", j / s; else print "0"}')
            fi ;;
        turbostat)
            stop_turbostat
            read -r power dram <<< "$(turbostat_avg)" ;;
        estimate)
            rm -f "$PID_FILE"
            wait $monitor_pid 2>/dev/null || true
            power=$(calculate_power "$TEMP_LOG") ;;
    esac
    
    # Extract results
    local throughput=$(extract_throughput "$BENCH_OUTPUT" "$workload_name")
    
    if [ -z "$throughput" ] || [ "$throughput" = "0" ]; then
        echo "Warning: Failed to extract throughput for $workload_name, threads=$threads" >&2
        throughput="0"
    fi
    
    # Calculate J/t (Joules per token)
    local j_per_token=$(awk -v p="$power" -v t="$throughput" 'BEGIN {
        if (t > 0) printf "%.4f", p/t; else print "0"
    }')
    
    # Output progress to stderr
    echo "  Throughput: $throughput t/s, Power: $power W (DRAM: $dram W, $POWER_USED), Energy: $j_per_token J/t" >&2
    
    # Only output CSV line to stdout (this will be captured)
    echo "$workload_name,$threads,$throughput,$power,$j_per_token,$POWER_USED,$dram"
}

# Initialize CSV
echo "Workload,Threads,Throughput(t/s),Power(W),Energy(J/t),PowerSource,DRAM(W)" > "$OUTPUT_CSV"

# Test PP workloads
IFS=',' read -ra PP_ARRAY <<< "$PP_THREADS"
for threads in "${PP_ARRAY[@]}"; do
    threads=$(echo "$threads" | xargs)  # trim whitespace
    result=$(run_benchmark "pp" "$threads")
    echo "$result" >> "$OUTPUT_CSV"
done

# Test TG workloads
IFS=',' read -ra TG_ARRAY <<< "$TG_THREADS"
for threads in "${TG_ARRAY[@]}"; do
    threads=$(echo "$threads" | xargs)  # trim whitespace
    result=$(run_benchmark "tg" "$threads")
    echo "$result" >> "$OUTPUT_CSV"
done

# Cleanup
rm -f "$TEMP_LOG" "$BENCH_OUTPUT" "$PID_FILE"  # the EXIT trap removes the turbostat temp dir

echo ""
echo "=== Benchmark Complete ==="
echo "Results saved to: $OUTPUT_CSV"
echo ""
cat "$OUTPUT_CSV"
