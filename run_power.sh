#!/bin/bash
# run_power.sh
# Runs every CNN model (batch sizes 2-2048) on CPU, GPU, and ANE while
# recording power with powermetrics.
#
# Runs under the normal user account. The ONLY command that needs admin
# rights is /usr/bin/powermetrics (read-only power readings), called with
# sudo below. Required sudoers rule:
#   sharm046 ALL=(root) NOPASSWD: /usr/bin/powermetrics
#
# Usage:  ./run_power.sh
# Takes about 10-15 minutes. Results go to the results/ folder:
#   time_<unit>_b<batch>.txt   timing + accuracy from bench
#   power_<unit>_b<batch>.txt  powermetrics samples during that run

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR" || exit 1
OUT="$DIR/results"
mkdir -p "$OUT"

# Check that powermetrics can run with sudo (no password prompt)
if ! sudo -n /usr/bin/powermetrics --samplers cpu_power -i 100 -n 1 > /dev/null 2>&1; then
    echo "Cannot run powermetrics with sudo. Admin access for /usr/bin/powermetrics is needed."
    exit 1
fi

DATA="digits_test.data"
TARGET_MS=10000    # keep each measured run going ~10 seconds, like the paper
INTERVAL_MS=200    # one power sample every 200 ms
UNITS="cpu gpu ane"
BATCHES="2 4 8 16 32 64 128 256 512 1024 2048"

echo "Started: $(date)"
for unit in $UNITS; do
    for b in $BATCHES; do
        model="CNNModels/CNN_Batch${b}_Inference.mlpackage"
        tag="${unit}_b${b}"
        echo "--- $tag ---"

        # 1. Quick 20-run timing to work out how many runs fill ~10 seconds
        avg=$(./bench "$model" "$DATA" "$b" "$unit" 20 | awk '/Average Runtime/ {print $3}')
        if [ -z "$avg" ]; then
            echo "  bench failed for $tag, skipping"
            continue
        fi
        runs=$(awk -v t="$TARGET_MS" -v a="$avg" 'BEGIN { r = int(t / a) + 1; if (r < 20) r = 20; print r }')

        # Enough power samples to cover: 2 s idle + benchmark + 4 s margin
        samples=$(awk -v r="$runs" -v a="$avg" -v i="$INTERVAL_MS" \
            'BEGIN { print int((2000 + r * a + 4000) / i) + 1 }')
        echo "  avg ${avg} ms -> ${runs} runs, ${samples} power samples"

        # 2. Start power recording (stops by itself after $samples readings)
        sudo -n /usr/bin/powermetrics --samplers cpu_power,gpu_power \
            -i "$INTERVAL_MS" -n "$samples" > "$OUT/power_${tag}.txt" 2>&1 &
        PM=$!
        sleep 2   # short idle baseline

        # 3. Measured run
        ./bench "$model" "$DATA" "$b" "$unit" "$runs" > "$OUT/time_${tag}.txt" 2>&1

        # 4. Wait for power recording to finish, then let the chip cool down
        wait "$PM"
        sleep 3
    done
done

echo "Finished: $(date)"
echo "Results saved in $OUT"
