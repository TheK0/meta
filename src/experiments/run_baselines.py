import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.training.train_meta import main as train_main

if __name__ == "__main__":
    # In a real scenario, this might set specific flags or load a specific config
    # For now, it just calls the main training entry point with the baseline config
    sys.argv = ["run_baselines.py", "--config", "configs/exp_baselines.yaml"]
    train_main()
