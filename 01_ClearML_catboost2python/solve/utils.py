import numpy as np
import torch
import os

def seed_everything(seed=2024):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def check_clearml_env(required_env_vars = None):
    if required_env_vars is None:
        required_env_vars = [
            "CLEARML_WEB_HOST",
            "CLEARML_API_HOST",
            "CLEARML_FILES_HOST",
            
            "CLEARML_API_ACCESS_KEY",
            "CLEARML_API_SECRET_KEY"
        ]

    missing_vars = [var for var in required_env_vars if os.getenv(var) is None]
    
    if missing_vars:
        print("⚠️  Некоторые переменные среды ClearML отсутствуют.")
        raise ValueError(f"Переменные среды ClearML отсутствуют: {missing_vars}")
