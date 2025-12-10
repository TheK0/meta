import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.training.train_meta import main as train_main

if __name__ == "__main__":
    sys.argv = ["run_ac_full.py", "--config", "configs/exp_ac_full.yaml"]
    train_main()
