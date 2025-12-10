import sys
import os
import argparse

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.datasets import MoleculePropertyDataset
from src.data.ac_precompute import ACPrecomputer

def main():
    parser = argparse.ArgumentParser(description="Precompute MMP and AC annotations.")
    parser.add_argument('--config', type=str, default='configs/datasets.yaml', help='Path to dataset config')
    parser.add_argument('--output', type=str, default='data/processed/ac_annotations.pkl', help='Output path')
    args = parser.parse_args()

    # Load config (dummy for now)
    config = {} 
    print(f"Loading configuration from {args.config}...")
    
    dataset = MoleculePropertyDataset(config)
    precomputer = ACPrecomputer(dataset)
    
    print("Starting MMP computation...")
    precomputer.compute_mmp_pairs()
    
    print("Labeling Activity Cliffs...")
    precomputer.label_ac_pairs()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    precomputer.save(args.output)
    print(f"AC annotations saved to {args.output}")

if __name__ == '__main__':
    main()
