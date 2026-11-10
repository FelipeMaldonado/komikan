#!/bin/zsh
# Self-play with late starts: half the games start near the end of a trewas win, going further back each generation.
# The wins come from late_wins.py, run first. Otherwise stage 3's control settings, so compare with runs/selfplay/control_*.
# Score with: python experiments/selfplay_evaluate.py --root runs/late --arms late --seeds 60 61 62 --no-finish
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
COMMON=(--generations 30 --games 200 --simulations 100 --reuse 4 --buffer 300000 --workers 3 --start-share 0
  --late runs/late/wins.json --late-share 0.5 --late-reach 10 200)
mkdir -p runs/late/logs
for seed in 60 61 62; do
  .venv/bin/python -m komikan.selfplay $COMMON --out runs/late/late_$seed --seed $seed > runs/late/logs/late_$seed.log 2>&1 &
done
wait
echo finished
