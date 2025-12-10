import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.training.train_meta import main as train_main

if __name__ == "__main__":
    sys.argv = ["run_ac_inner.py", "--config", "configs/exp_ac_inner.yaml"]
    train_main()
