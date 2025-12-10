import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error

def safe_roc_auc(y_true, y_score):
    y_true = np.asarray(y_true)
    # 如果只有一个类别，直接返回 NaN（或者 0.5 随机水平，看你需求）
    if np.unique(y_true).size < 2:
        return float('nan')
    return roc_auc_score(y_true, y_score)

class Evaluator:
    """
    Evaluator for classification and regression metrics.
    """
    def __init__(self, task_type='classification'):
        self.task_type = task_type

    def compute(self, y_true, y_pred, ac_mask=None):
        """
        Compute metrics.
        
        Args:
            y_true: True labels.
            y_pred: Predicted scores/values.
            ac_mask: Boolean mask indicating which samples are Activity Cliffs (optional).
        """
        metrics = {}
        
        # 1. Global Metrics
        if self.task_type == 'classification':
            try:
                metrics['roc_auc'] = safe_roc_auc(y_true, y_pred)
            except ValueError:
                metrics['roc_auc'] = 0.5
        elif self.task_type == 'regression':
            metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            
        # 2. AC-Specific Metrics (if mask provided)
        if ac_mask is not None and np.sum(ac_mask) > 0:
            y_true_ac = np.array(y_true)[ac_mask]
            y_pred_ac = np.array(y_pred)[ac_mask]
            
            if len(y_true_ac) > 0:
                if self.task_type == 'classification':
                    try:
                        metrics['roc_auc_ac'] = safe_roc_auc(y_true_ac, y_pred_ac)
                    except ValueError:
                        metrics['roc_auc_ac'] = 0.5
                elif self.task_type == 'regression':
                    metrics['rmse_ac'] = np.sqrt(mean_squared_error(y_true_ac, y_pred_ac))
                    
        return metrics
