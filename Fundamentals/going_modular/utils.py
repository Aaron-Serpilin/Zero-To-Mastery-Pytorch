""" 
Contains various utility functions for PyTorch model training and saving
"""

import torch
from pathlib import Path

def save_model(
    model: torch.nn.Module, # model to save
    target_dir: str, # directory for saving the model to
    model_name: str # filename for the saved model. Should include either ".pth" or ".pt" as the file extension
    ):

    """ 
    Saves a PyTorch model to a target directory
    """

    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model.name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)