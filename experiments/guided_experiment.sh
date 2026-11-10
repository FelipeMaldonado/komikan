#!/bin/zsh
# Stage 2: does guidance from the existing networks beat plain search at the same budget?
# Settings come from the pilot (runs/guided/pilot.txt). This uses other opponents and seeds. The value networks must be exported first:
#   .venv/bin/python -c "from komikan.puct import export_value_net as e; [e(f'models/{n}.zip', f'runs/guided/nets/{n}_value.npz') for n in ('komikelu','trewas')]"
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
run() { .venv/bin/python experiments/guided_evaluate.py "$@"; }
TREWAS=(mcts:400:0.5:2 mcts:1600:0.5:2 puct:400:prior puct:400:both puct:1600:prior puct:1600:both)
KOMIKELU=(mcts:400:0:2 mcts:1600:0:2 puct:400:both puct:1600:both)
{
  run --set test --role trewas --games 40 --specs $TREWAS
  run --set test --role komikelu --games 40 --specs $KOMIKELU
  run --finish --plays 2 --specs mcts:1600:0.5:2 puct:400:prior puct:1600:prior puct:1600:both
} > runs/guided/results.txt 2>&1
echo finished
