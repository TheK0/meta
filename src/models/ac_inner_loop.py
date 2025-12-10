import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ACInnerLoop(nn.Module):
    """
    AC-aware Inner Loop Optimization.
    Implements weighted loss updates based on Activity Cliff (AC) information
    and dynamic curriculum (Top-R% selection).
    """
    def __init__(self, config):
        super(ACInnerLoop, self).__init__()
        self.lr_inner = config.get('inner_lr', 0.01)
        self.ac_weight = config.get('ac_weight', 2.0) # Weight p_i for AC samples
        self.r_start = config.get('r_start', 1.0) # Initial ratio of samples to keep
        self.r_end = config.get('r_end', 0.5)     # Final ratio
        self.total_steps = config.get('total_steps', 10000) # For curriculum schedule

    def get_current_ratio(self, global_step):
        """Calculate current keep ratio R(t)."""
        progress = min(1.0, global_step / self.total_steps)
        return self.r_start + progress * (self.r_end - self.r_start)

    def step(self, model, support_logits, support_labels, ac_mask, global_step):
        """
        Perform one step of inner loop adaptation with AC-aware weighting.
        
        Args:
            model: The model to update.
            support_logits: Predictions on support set.
            support_labels: True labels.
            ac_mask: Boolean tensor indicating AC samples.
            global_step: Current global training step.
            
        Returns:
            updated_params: Dict of updated parameters (fast weights).
            stats: Dict of AC statistics.
        """
        # 1. Compute per-sample loss
        loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        per_sample_loss = loss_fn(support_logits.view(-1), support_labels)
        
        # 2. Apply AC Weights (p_i)
        # w_i = p_i if is_AC else 1.0
        weights = torch.ones_like(per_sample_loss)
        if ac_mask is not None:
            weights[ac_mask] = self.ac_weight
            
        weighted_loss = per_sample_loss * weights
        
        # 3. Top-R(t)% Selection (Curriculum)
        current_r = self.get_current_ratio(global_step)
        k = int(len(weighted_loss) * current_r)
        k = max(1, k) # Keep at least 1
        
        # Sort by weighted loss (hardest examples first)
        top_k_loss, top_k_indices = torch.topk(weighted_loss, k)
        
        # Final Loss for update
        final_loss = top_k_loss.mean()
        
        # 4. Gradient Update
        grads = torch.autograd.grad(final_loss, model.parameters(), create_graph=True, allow_unused=True)
        
        updated_params = {}
        for (name, param), grad in zip(model.named_parameters(), grads):
            if grad is not None:
                updated_params[name] = param - self.lr_inner * grad
            else:
                updated_params[name] = param
                
        # 5. Statistics
        stats = {
            'inner_loss': final_loss.item(),
            'ac_ratio': ac_mask.float().mean().item() if ac_mask is not None else 0.0,
            'keep_ratio': current_r
        }
        
        return updated_params, stats
