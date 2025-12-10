def get_sweep_config():
    """
    Return a configuration for hyperparameter sweep (e.g. for WandB or Optuna).
    """
    return {
        'method': 'bayes',
        'metric': {'name': 'val_loss', 'goal': 'minimize'},
        'parameters': {
            'lr': {'min': 0.0001, 'max': 0.01},
            'hidden_dim': {'values': [64, 128, 256]}
        }
    }
