#!/bin/zsh
# Self-play under the forward-or-sideways variant, with stage 3's control settings (no human starts, which come from standard games).
# Compare its self-play outcomes with runs/selfplay/logs/control_*.log, then score with variant_evaluate.py.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
COMMON=(--generations 30 --games 200 --simulations 100 --reuse 4 --buffer 300000 --workers 3 --start-share 0 --rules no-backward)
mkdir -p runs/variant/logs
for seed in 50 51 52; do
  .venv/bin/python -m komikan.selfplay $COMMON --out runs/variant/selfplay_$seed --seed $seed > runs/variant/logs/selfplay_$seed.log 2>&1 &
done
wait
echo finished
