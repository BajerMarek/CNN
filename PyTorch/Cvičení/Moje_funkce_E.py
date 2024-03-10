import torch
from torch import nn
#! Torchvision
import torchvision
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
#! Mathplotlib
import matplotlib.pyplot as plt
#! Další
from timeit import default_timer as timer
from tqdm.auto import tqdm



#! ******************* Moje vlastní dílo ***************************
device = "cuda" if torch.cuda.is_available() else "cpu"
def train_step(model:torch.nn.Module,
               data_loader:torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module,
               optimizer:torch.optim.Optimizer,
               accuracy_fn,
               device:torch.device=device):
    """Vytvoří trainig loop pro zadaný model"""
    train_loss, train_acc = 0,0
    model.train()
    #! Trainig loop
    for batch, (X, y) in enumerate(data_loader):
        #? Převod dat na správné zařízení
        X,y = X.to(device),y.to(device)
        #? Forward pass
        y_pred = model(X)
        #? Loss per bach
        loss = loss_fn(y_pred,y)
        train_loss += loss      #? přidává do promněné 
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1))
        #? Optimize zero grad
        optimizer.zero_grad()
        #? Back propagation
        loss.backward()
        #? Optimizer step
        optimizer.step()
        if batch % 400 == 0:
            print(f"Looked at {batch*len(X)}/{len(data_loader.dataset)} sampels.")

    #? Dělení train_loss  dékou dataloaderu(batches)
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    
    
#! Test loop (ve formně funkce)
def test_step(model:torch.nn.Module,
              data_loader:torch.utils.data.DataLoader,
              loss_fn:torch.nn.Module,
              accuracy_fn,
              device:torch.device = device):
    test_loss,test_acc = 0,0
    model.eval()
    for X_test,y_test in data_loader:
        #! Převod dat na správné zařízení
        X_test,y_test = X_test.to(device),y_test.to(device)
        #? Forward pass
        test_pred = model(X_test)                              #!
        #? Test loss
        test_loss += loss_fn(test_pred,y_test)                    #!
        #? Přesnost
        test_acc += accuracy_fn(y_true=y_test,y_pred=test_pred.argmax(dim=1))   #!
        #? Průměená test loss
    test_loss /= len(data_loader)
    test_acc /= len(data_loader)
    #! Vizualizace
    print(f"\nTest loss: {test_loss:.5f} | Test acc: {test_acc:.5f}%")

def eval_model(model:torch.nn.Module,
               data_loader:torch.utils.data.DataLoader, # test dataloader
               loss_fn:torch.nn.Module,
               accuracy_fn):
    """ Vrací dictonary s výsledky modelu
     dataloader -> test_dataloader """
    loss,acc = 0,0
    model.eval()
    with torch.inference_mode():
        for X,y in tqdm(data_loader):
            X,y = X.to(device),y.to(device)
            y_pred = model(X)
            loss += loss_fn(y_pred,y)
            acc += accuracy_fn(y_true=y,
                               y_pred=y_pred.argmax(dim=1))
        loss /= len(data_loader)
        acc /= len(data_loader)
    return {"model_name":model.__class__.__name__,
            "model_loss":loss.item(),
            "model_acc":acc}

def make_predictions(model:torch.nn.Module,
                     data:list,
                     device:torch.device=device):
    pred_probs = []
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for sample in data:
            #? příprava dat (přidá dimenzu probatch a převede na správné device)
            sample=torch.unsqueeze(sample,dim=0).to(device)
            #? Forward pass
            pred_logits = model(sample)

            #? predicrion propability
            pred_prob = torch.softmax(pred_logits.squeeze(),dim=0)

            #? pred prob na cpu
            pred_probs.append(pred_prob.cpu())

#? Naskládaní listu do tenzoru
    return torch.stack(pred_probs)

def print_train_time(start: float,
                     end: float,
                     device: torch.device = None):
    """ Vypíše čas od začátku do konce"""
    total_time = end - start,
    print(f"Train time on {device}: {total_time} sekund")
    return total_time
