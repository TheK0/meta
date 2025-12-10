import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import functional_call
import numpy as np
from collections import defaultdict
from src.data.featurization import smiles_to_graph
from torch_geometric.data import Batch, Data


def batch_graphs(smiles_list, labels, mids, device):
    """
    Unified graph batching utility.
    
    Args:
        smiles_list: List of SMILES strings.
        labels: List of labels.
        mids: List of molecule IDs (for MPG mapping).
        device: Target device.
        
    Returns:
        PyG Batch object with mids attached, or None if all graphs fail.
    """
    data_list = []
    valid_mids = []
    
    for s, y, mid in zip(smiles_list, labels, mids):
        g = smiles_to_graph(s)
        if g:
            data = Data(
                x=torch.tensor(g['node_features'], dtype=torch.float),
                edge_index=torch.tensor(g['edge_index'], dtype=torch.long),
                edge_attr=torch.tensor(g['edge_features'], dtype=torch.float),
                y=torch.tensor([y], dtype=torch.float)
            )
            data_list.append(data)
            valid_mids.append(mid)
    
    if not data_list:
        return None
    
    batch = Batch.from_data_list(data_list).to(device)
    batch.mids = valid_mids  # Attach molecule IDs for MPG mapping
    return batch


def meta_train_loop(model, dataset, config, ac_inner=None, scheduler=None):
    """
    Meta-training loop with AC-aware inner loop.
    
    Args:
        model: GSMetaCore model.
        dataset: MoleculePropertyDataset.
        config: Configuration dict.
        ac_inner: Optional ACInnerLoop instance.
        scheduler: Optional ACScheduler instance (for future use).
    """
    optimizer = optim.Adam(model.parameters(), lr=config.get('meta_lr', 0.001))
    epochs = config.get('epochs', 10)
    n_tasks_per_epoch = config.get('n_tasks_per_epoch', 10)
    k_support = config.get('k_shot', 5)
    k_query = config.get('q_query', 5)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    global_step = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        tasks_processed = 0
        
        for t in range(n_tasks_per_epoch):
            # 1. Sample Task
            task_sample = dataset.sample_task('train', k_support, k_query)
            if task_sample is None:
                continue
            
            prop_id, sup_mids, q_mids, sup_labels, q_labels, sup_ac_mask, q_ac_mask = task_sample
            
            # 2. Featurize
            sup_smiles = [dataset.get_molecule_smiles(m) for m in sup_mids]
            q_smiles = [dataset.get_molecule_smiles(m) for m in q_mids]
            
            sup_batch = batch_graphs(sup_smiles, sup_labels, sup_mids, device)
            q_batch = batch_graphs(q_smiles, q_labels, q_mids, device)
            
            if sup_batch is None or q_batch is None:
                continue
            
            # 3. Inner Loop: Update Classifier on Support Set
            # Freeze GNN encoder
            with torch.no_grad():
                sup_emb = model.mol_encoder(sup_batch.x, sup_batch.edge_index, sup_batch.batch)
            
            # Forward Classifier (detach embeddings to prevent GNN gradients)
            sup_logits = model.classifier(sup_emb.detach())
            
            if ac_inner is not None:
                # AC-Aware Inner Loop
                sup_ac_tensor = torch.tensor(sup_ac_mask, dtype=torch.bool, device=device)
                updated_params, stats = ac_inner.step(
                    model.classifier,
                    sup_logits,
                    sup_batch.y.view(-1),
                    sup_ac_tensor,
                    global_step
                )
            else:
                # Standard Inner Loop (one GD step)
                loss_fn = nn.BCEWithLogitsLoss()
                sup_loss = loss_fn(sup_logits.view(-1), sup_batch.y.view(-1))
                
                grads = torch.autograd.grad(sup_loss, model.classifier.parameters(), create_graph=True)
                updated_params = {}
                lr_inner = config.get('inner_lr', 0.01)
                for (name, param), grad in zip(model.classifier.named_parameters(), grads):
                    updated_params[name] = param - lr_inner * grad
            
            # 4. Outer Loop: Evaluate on Query Set with Fast Weights
            with torch.no_grad():
                q_emb = model.mol_encoder(q_batch.x, q_batch.edge_index, q_batch.batch)
            
            # Use functional_call to apply fast weights
            q_logits = functional_call(model.classifier, updated_params, (q_emb,))
            
            loss_fn = nn.BCEWithLogitsLoss()
            q_loss = loss_fn(q_logits.view(-1), q_batch.y.view(-1))
            
            # 5. Backpropagate to Global Parameters
            optimizer.zero_grad()
            q_loss.backward()
            optimizer.step()
            
            total_loss += q_loss.item()
            tasks_processed += 1
            global_step += 1
        
        avg_loss = total_loss / max(tasks_processed, 1)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")


def meta_test_loop(model, dataset, config, evaluator, device):
    """
    Meta-testing loop with test-time adaptation.
    
    Args:
        model: GSMetaCore model.
        dataset: MoleculePropertyDataset.
        config: Configuration dict.
        evaluator: Evaluator instance.
        device: Device.
    """
    model.eval()
    n_test_tasks = config.get('n_test_tasks', 100)
    k_support = config.get('k_shot', 5)
    k_query = config.get('q_query', 5)
    
    print(f"\nStarting Meta-Testing on {n_test_tasks} tasks...")
    
    results = defaultdict(list)
    
    for i in range(n_test_tasks):
        # 1. Sample Task (Test Split)
        task_sample = dataset.sample_task('test', k_support, k_query)
        if task_sample is None:
            continue
        
        prop_id, sup_mids, q_mids, sup_labels, q_labels, sup_ac_mask, q_ac_mask = task_sample
        
        # 2. Featurize
        sup_smiles = [dataset.get_molecule_smiles(m) for m in sup_mids]
        q_smiles = [dataset.get_molecule_smiles(m) for m in q_mids]
        
        sup_batch = batch_graphs(sup_smiles, sup_labels, sup_mids, device)
        q_batch = batch_graphs(q_smiles, q_labels, q_mids, device)
        
        if sup_batch is None or q_batch is None:
            continue
        
        # 3. Test-Time Adaptation (Inner Loop)
        with torch.no_grad():
            sup_emb = model.mol_encoder(sup_batch.x, sup_batch.edge_index, sup_batch.batch)
        
        with torch.enable_grad():
            # Forward Classifier (detached embeddings)
            sup_logits = model.classifier(sup_emb.detach())
            sup_labels_tensor = sup_batch.y.view(-1)
            
            loss_fn = nn.BCEWithLogitsLoss()
            sup_loss = loss_fn(sup_logits.view(-1), sup_labels_tensor)
            
            # Gradient-based adaptation
            grads = torch.autograd.grad(sup_loss, model.classifier.parameters())
            updated_params = {}
            lr_inner = config.get('inner_lr', 0.01)
            for (name, param), grad in zip(model.classifier.named_parameters(), grads):
                updated_params[name] = param - lr_inner * grad
        
        # 4. Evaluate on Query Set
        with torch.no_grad():
            q_emb = model.mol_encoder(q_batch.x, q_batch.edge_index, q_batch.batch)
            
            q_logits = functional_call(model.classifier, updated_params, (q_emb,))
            q_probs = torch.sigmoid(q_logits).view(-1).cpu().numpy()
            q_labels_np = q_batch.y.cpu().numpy()
            q_ac_mask_np = np.array(q_ac_mask, dtype=bool)
            
            # Compute Metrics
            task_metrics = evaluator.compute(q_labels_np, q_probs, q_ac_mask_np)
            for k, v in task_metrics.items():
                results[k].append(v)
    
    # 5. Aggregate Results
    print("\nMeta-Testing Results:")

    for k, v in results.items():
        v = np.array(v, dtype=float)
        # 使用 nanmean/nanstd 避免因 NaN 导致结果全崩
        mean = np.nanmean(v)
        std = np.nanstd(v)
        print(f"{k}: {mean:.4f} +/- {std:.4f}")

    return results

