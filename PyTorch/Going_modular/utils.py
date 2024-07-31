import torch            #! 21 48
from pathlib import Path
from torch import nn
import matplotlib.pyplot as plt

def save_model(model:torch.nn.Module,
               target_dir:str,
               model_name:str):
    """Uloží model do dané složky s danným jménem"""
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)
    
    assert model_name.endswith(".pth") or model_name.endswith(".pt"),"Koncovka modelu musí být .pth nebo .pt"
    model_save_path = target_dir_path / model_name
    print(f"[INFO] stahuji model do {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)

def plot_loss_curves(results:dict[str,list[float]]):
    
    #? získání dat
    loss = results["train_loss"]
    test_loss = results["test_loss"]
    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]
    epochs=range(len(results["train_loss"]))
    #? zobrazení
    plt.figure(figsize=(15,7))
    #? loss
    plt.subplot(1,2,1)
    plt.plot(epochs,loss,label="train_loss")
    plt.plot(epochs,test_loss,label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    #? acc
    plt.subplot(1,2,2)
    plt.plot(epochs,accuracy,label="train_accuracy")
    plt.plot(epochs,test_accuracy,label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()
            

