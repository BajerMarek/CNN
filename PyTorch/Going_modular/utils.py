import torch            #! 21 48
from pathlib import Path
from torch import nn


def save_model(model:torch.nn.Module,
               target_dir:str,
               model_name:str):
    """Uloží model do dané složky s danným jménem"""
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)
    
    assert model.endswith(".pth") or model_name.endswith(".pt"),"Koncovka modelu musí bít .pth nebo .pt"
    model_save_path = target_dir_path / model_name
    print(f"[INFO] stahuji model do {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)
               
                
    
    