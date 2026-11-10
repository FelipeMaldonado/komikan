#!/bin/zsh
# Search with no network: strength by budget, in both roles, and finishing from the human's positions.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
mkdir -p runs/mcts
run() { .venv/bin/python experiments/mcts_evaluate.py --workers 8 "$@"; }
{
  run --games 40 --simulations 100 400 1600 --draws 0.5 --playouts 2
  run --games 40 --simulations 400 --draws 0 --playouts 2 --roles trewas
  run --finish --simulations 100 400 1600 --draws 0.5 --playouts 2 --plays 2
  run --games 16 --simulations 6400 --draws 0.5 --playouts 2 --only random "minimax 1" "minimax 2" "K2 models" "K1 best" "K1 gen 11" "T2 models" "T1 best"
} > runs/mcts/results.txt 2>&1
echo finished
