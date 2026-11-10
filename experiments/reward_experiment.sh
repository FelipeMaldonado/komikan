#!/bin/zsh
# Does paying for confinement, instead of for marching to the triangle, fix the hiding the human spotted?
# All groups train from scratch with the same partners, none of which is used in the final test's "unseen" columns.
#   current: today's rewards.  harsh_draw: a draw costs as much as a loss.
#   confine: no advance bonus, four times the confinement bonus.  relative: as confine, plus advance pays only up to the komikelu's row.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
S=runs/second/snapshots
COMMON=(--no-clone --warmup-steps 200000 --generations 6 --steps-per-generation 300000 --envs 2 --eval-games 20
  --partners $S/komikelu_002.npz $S/komikelu_008.npz $S/komikelu_011.npz
  --benchmarks random minimax:1 minimax:2 policy:$S/komikelu_005.npz)
mkdir -p runs/reward/logs
train() { .venv/bin/komikan-pretrain $COMMON --seed $2 --out runs/reward/$1_$2 "${@:3}" > runs/reward/logs/$1_$2.log 2>&1 & }
for seed in 30 31 32; do
  train current $seed
  train harsh_draw $seed --reward-draw-trewas -1
  train confine $seed --reward-advance 0 --reward-confine 2
  train relative $seed --relative-advance --reward-confine 2
done
wait
echo finished
