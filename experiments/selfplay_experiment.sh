#!/bin/zsh
# Stage 3: small self-play training from scratch, with no shaped rewards and no PPO agents as partners.
#   human: a quarter of games start from positions in the human's wins, with one win held out per seed.  control: all games from the start.
# None of the opponents in selfplay_evaluate.py is used in training.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
COMMON=(--generations 30 --games 200 --simulations 100 --reuse 4 --buffer 300000 --workers 2)
mkdir -p runs/selfplay/logs
train() { .venv/bin/python -m komikan.selfplay $COMMON --out runs/selfplay/$1_$2 --seed $2 "${@:3}" > runs/selfplay/logs/$1_$2.log 2>&1 & }
for i in 0 1 2; do
  seed=$((40 + i))
  train human $seed --start-share 0.25 --hold-out $i
  train control $seed --start-share 0
done
wait
echo finished
