import torch
import torch.utils
import torch.utils.data
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model:torch.nn.Module,
               loss_fn:torch.nn.Module,
               optimizer:torch.optim.Optimizer,
               dataloader:torch.utils.data.DataLoader,
               device:torch.device -> Tuple[float,float]):
    """Vytvoří trainig loop která buve cviřit model pro jednu epochu"""
