#!/bin/bash
# Run all experiments in order. Log errors, continue on failure.
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
mkdir -p results/logs results/tables results/figures

python experiments/exp1_main.py 2>&1 | tee results/logs/exp1.log || true
echo "EXP1 DONE"
python experiments/exp2_shots.py 2>&1 | tee results/logs/exp2.log || true
echo "EXP2 DONE"
python experiments/exp3_federated.py 2>&1 | tee results/logs/exp3.log || true
echo "EXP3 DONE"
python experiments/exp4_ablation.py 2>&1 | tee results/logs/exp4.log || true
echo "EXP4 DONE"
python experiments/exp5_generalization.py 2>&1 | tee results/logs/exp5.log || true
echo "EXP5 DONE"

echo "All experiments complete. Run: python results/generate_all_outputs.py"
