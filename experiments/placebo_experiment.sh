#!/bin/zsh
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
S=runs/second/snapshots
COMMON=(--warmup-steps 150000 --generations 6 --steps-per-generation 150000 --envs 2 --eval-games 20
  --learning-rate 2e-4 --ent-coef 0.005 --init models/trewas.zip
  --partners $S/komikelu_002.npz $S/komikelu_008.npz $S/komikelu_011.npz
  --benchmarks random minimax:1 minimax:2 policy:$S/komikelu_005.npz)
CLONE=(--clone-epochs 10 --clone-learning-rate 1e-4 --recordings recordings/*.json)
mkdir -p runs/bc2/logs
for seed in 10 11 12; do
  .venv/bin/komikan-pretrain $COMMON --seed $seed --no-clone --out runs/bc2/control_$seed > runs/bc2/logs/control_$seed.log 2>&1 &
  .venv/bin/komikan-pretrain $COMMON $CLONE --seed $seed --out runs/bc2/human_$seed > runs/bc2/logs/human_$seed.log 2>&1 &
  .venv/bin/komikan-pretrain $COMMON $CLONE --placebo --seed $seed --out runs/bc2/placebo_$seed > runs/bc2/logs/placebo_$seed.log 2>&1 &
done
wait
echo finished
