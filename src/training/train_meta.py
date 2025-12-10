import os
import sys
# 将项目根目录添加到 Python 路径
# 获取当前文件路径: ML/meta/src/training/train_meta.py
current_file = os.path.abspath(__file__)
# 向上四级到达 meta 目录
meta_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
# 将 meta 目录添加到 sys.path
if meta_dir not in sys.path:
    sys.path.insert(0, meta_dir)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import yaml
import torch
import pickle

from src.data.datasets import MoleculePropertyDataset
from src.data.ac_precompute import ACPrecomputer
from src.models.gsmeta_core import GSMetaCore
from src.models.ac_inner_loop import ACInnerLoop
from src.training.loops import meta_train_loop, meta_test_loop
from src.training.evaluators import Evaluator
from src.training.utils_seed import set_seed

def main():
    parser = argparse.ArgumentParser(description="Meta-learning training entry point.")
    parser.add_argument('--config', type=str, default='configs/model_base.yaml', help='Path to config file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--precompute_ac', action='store_true', help='Run AC precomputation before training')
    args = parser.parse_args()

    # === 处理配置文件路径 ===
    config_path = args.config
    
    # 如果是相对路径且文件不存在，尝试在 meta 目录下查找
    if not os.path.isabs(config_path) and not os.path.exists(config_path):
        # meta_dir 已经在文件开头定义
        alt_config_path = os.path.join(meta_dir, config_path)
        
        if os.path.exists(alt_config_path):
            config_path = alt_config_path
            print(f"✓ Loading config from: {config_path}")
        else:
            print(f"✗ Error: Cannot find config file!")
            print(f"  Tried locations:")
            print(f"    - {os.path.abspath(args.config)}")
            print(f"    - {alt_config_path}")
            print(f"\n  Suggestion: Run with --config meta/configs/model_base.yaml")
            sys.exit(1)
    else:
        print(f"✓ Loading config from: {config_path}")

    # 1. Load Config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override seed if provided
    if args.seed is not None:
        config['seed'] = args.seed
    
    set_seed(config.get('seed', 42))
    
    # 2. Initialize Dataset
    print("Loading dataset...")
    dataset = MoleculePropertyDataset(config)
    
    # 3. AC Precomputation (if needed)
    ac_annotations_path = r"C:\Users\yishengyuan\Downloads\ML\meta\src\data\processed\ac_annotations.pkl"
    
    if args.precompute_ac or not os.path.exists(ac_annotations_path):
        print("Running AC precomputation on train split...")
        ac_precomputer = ACPrecomputer(dataset)
        
        # Step 1: Compute MMP pairs (structural analogs)
        ac_precomputer.compute_mmp_pairs()
        
        # Step 2: Label AC pairs (only on train split to prevent leakage)
        ac_precomputer.label_ac_pairs(
            threshold=config.get('ac_threshold', 0.0),
            split='train'
        )
        
        # Step 3: Extract AC node indices from is_ac_node dict
        ac_node_indices = {mid for mid, is_ac in ac_precomputer.is_ac_node.items() if is_ac}
        
        # Save annotations
        os.makedirs('data/processed', exist_ok=True)
        with open(ac_annotations_path, 'wb') as f:
            pickle.dump({
                'ac_node_indices': ac_node_indices,
                'ac_pairs': ac_precomputer.ac_pairs,
                'mmp_pairs': ac_precomputer.mmp_pairs
            }, f)
        
        print(f"AC precomputation complete. Found {len(ac_node_indices)} AC nodes.")
    else:
        print(f"Loading existing AC annotations from {ac_annotations_path}...")
        with open(ac_annotations_path, 'rb') as f:
            ac_data = pickle.load(f)
            ac_node_indices = ac_data['ac_node_indices']
        print(f"Loaded {len(ac_node_indices)} AC nodes.")
    
    # Set AC annotations in dataset
    dataset.set_ac_annotations(ac_node_indices)
    
    # 4. Initialize Model
    print("Initializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GSMetaCore(config)
    model.to(device)
    
    # 5. Initialize AC Inner Loop (if enabled)
    ac_inner = None
    if config.get('use_ac_inner_loop', False):
        print("Initializing AC-Aware Inner Loop...")
        ac_inner = ACInnerLoop(config)
    
    # 6. Initialize Evaluator
    task_type = config.get('task_type', 'classification')
    evaluator = Evaluator(task_type=task_type)
    
    # 7. Meta-Training
    print("\n" + "="*50)
    print("Starting Meta-Training...")
    print("="*50)
    meta_train_loop(model, dataset, config, ac_inner=ac_inner)
    
    # 8. Save Checkpoint
    checkpoint_path = config.get('checkpoint_path', 'checkpoints/model_final.pt')
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nModel saved to {checkpoint_path}")
    
    # 9. Meta-Testing
    print("\n" + "="*50)
    print("Starting Meta-Testing...")
    print("="*50)
    results = meta_test_loop(model, dataset, config, evaluator, device)
    
    # 10. Save Results
    results_path = config.get('results_path', 'results/meta_test_results.pkl')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
