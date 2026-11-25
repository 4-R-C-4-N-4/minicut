#!/usr/bin/env bash
# Seed sweep: v1 pilot data vs verified v2, seeds 1..3, interleaved so a
# partial sweep still compares like with like. Runs sequentially on the 3090.
set -u
cd /home/ivy/Work/minicut
BASE="$HOME/.cache/huggingface/hub/models--openbmb--MiniCPM5-2B/snapshots/42433e90dad95123ab87729fd7011e7110629556"
PY=$HOME/Work/rellm/.venv/bin/python
LOG=$TMPDIR/sweep.log
: >> "$LOG"
for seed in 1 2 3; do
  for ds in ${DATASETS:-pilot_v1 verified_v2}; do
    run=training/output/sweep-${ds}-s${seed}
    if [ -f "$run/comparison.json" ]; then echo "skip $run (done)" >> "$LOG"; continue; fi
    rm -rf "$run"
    echo "$(date +%T) start $run" >> "$LOG"
    if systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=1G \
        $PY -m training.factory pilot --run "$run" --base "$BASE" \
        --gguf "$HOME/programs/openbmb/MiniCPM5-2B/MiniCPM5-2B-Q4_K_M.gguf" \
        --llama "$HOME/programs/llama.cpp" --gpu GPU-1e207c4c-9743-52ff-2165-b19430f5c2ae \
        --data "training/train_${ds}.jsonl" --seed "$seed" > "$run.out" 2>&1; then
      echo "$(date +%T) done $run: $(tail -c 2000 $run.out | tr -d '\n ' | grep -o '"stock":[0-9]*,"tuned":[0-9]*,"gains":[0-9]*,"regressions":[0-9]*,"gate_pass":[a-z]*')" >> "$LOG"
    else
      echo "$(date +%T) FAILED $run (see $run.out)" >> "$LOG"
    fi
  done
done
echo "$(date +%T) sweep finished" >> "$LOG"
