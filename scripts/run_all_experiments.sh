#!/bin/bash

echo "Running Baselines..."
python src/experiments/run_baselines.py

echo "Running AC Inner Loop..."
python src/experiments/run_ac_inner.py

echo "Running AC Full..."
python src/experiments/run_ac_full.py

echo "Running Ablation Studies..."
python src/experiments/run_ablation.py

echo "All experiments completed."
