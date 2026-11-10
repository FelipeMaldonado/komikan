#!/bin/zsh
# Does starting from mid-game positions in the human's wins help? Three groups, same settings as the placebo experiment:
#   control: no mid-game starts.  agent: starts from ordinary agent games, at the same turn numbers.  human: starts from the wins.
# Each seed holds out one recorded game, so midgame_evaluate.py can measure finishing from unseen positions.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
S=runs/second/snapshots
COMMON=(--no-clone --warmup-steps 150000 --generations 6 --steps-per-generation 150000 --envs 2 --eval-games 20
  --learning-rate 2e-4 --ent-coef 0.005 --init models/trewas.zip
  --partners $S/komikelu_002.npz $S/komikelu_008.npz $S/komikelu_011.npz
  --benchmarks random minimax:1 minimax:2 policy:$S/komikelu_005.npz)
STARTS=(--recordings recordings/*.json --start-share 0.5 --start-min-progress 0.25)
mkdir -p runs/midgame/logs
for held_out in 0 1 2 3 4; do
  seed=$((20 + held_out))
  .venv/bin/komikan-pretrain $COMMON --seed $seed --out runs/midgame/control_$seed > runs/midgame/logs/control_$seed.log 2>&1 &
  .venv/bin/komikan-pretrain $COMMON $STARTS --start-from agent --start-holdout $held_out --seed $seed --out runs/midgame/agent_$seed > runs/midgame/logs/agent_$seed.log 2>&1 &
  .venv/bin/komikan-pretrain $COMMON $STARTS --start-from human --start-holdout $held_out --seed $seed --out runs/midgame/human_$seed > runs/midgame/logs/human_$seed.log 2>&1 &
done
wait
echo finished
