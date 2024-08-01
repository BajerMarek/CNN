import torch                #! 17 22
import torch.utils
import torch.utils.data
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model:torch.nn.Module,
               loss_fn:torch.nn.Module,
               optimizer:torch.optim.Optimizer,
               dataloader:torch.utils.data.DataLoader,
               device:torch.device): #-> Tuple[float,float]):
    """Vytvoří trainig loop která buve cviřit model pro jednu epochu
            vrátí tuple ve formatu (trainig_loss, trainig_eval_metric)
            eval_metric -> způsob ohodnocení modelu např. přesnost"""
    
    model.train()
    train_loss, train_acc = 0,0
    for batch,(X,y) in enumerate(dataloader):
        X,y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred,y)
        train_loss += loss.item()

        optimizer.zero_grad()
        
        loss.backward()

        optimizer.step()
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    train_loss = train_loss/len(dataloader)
    train_acc = train_acc/len(dataloader)
    return train_loss, train_acc

def test_step(model:torch.nn.Module,
              loss_fn:torch.nn.Module,
              dataloader:torch.utils.data.DataLoader,
              device:torch.device): #-> Tuple[float,float]):
    """Funkce která vytvoří testovací loop ve keterném se model ohodnotí
            Výsledek bude tuple ve formatu (test_loss,test_eval_metric)
            eval_metric -> způsob ohodnocení modelu např. přesnost"""
    test_loss, test_acc =0,0
    model.eval()
    with torch.inference_mode():
        for batch,(X,y) in enumerate(dataloader):
            X,y = X.to(device), y.to(device)

            y_test_pred = model(X)
            loss = loss_fn(y_test_pred,y)
            test_loss += loss.item()

            test_pred_labels = y_test_pred.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item()/len(test_pred_labels)
    test_loss = test_loss/len(dataloader)
    test_acc = test_acc/len(dataloader)
    return test_loss, test_acc

def train(model:torch.nn.Module,
          train_dataloader:torch.utils.data.DataLoader,
          test_dataloadeer:torch.utils.data.DataLoader,
          loss_fn:torch.nn.Module,
          optimizer:torch.optim.Optimizer,
          epochs:int,
          device:torch.device):
    """Tato funkce kombinuje train_step a test_step a dohromady učí model
            Výsledek:{train_loss: [...],
                  train_acc: [...],
                  test_loss: [...],
                  test_acc: [...]}"""
    
    results = {"train_loss":[],
               "train_acc":[],
               "test_loss":[],
               "test_acc":[]}
    for epochs in tqdm(range(epochs)):
         train_loss, train_acc = train_step(model=model,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer,
                                            dataloader=train_dataloader,
                                            device=device)
         test_loss, test_acc = test_step(model=model,
                                         loss_fn=loss_fn,
                                         dataloader=test_dataloadeer,
                                         device=device)
         
         print(f"Epocha: {epochs+1} | Train_loss: {train_loss} | Test_loss: {test_loss} | Train_acc: {train_acc} | Test_acc: {test_acc}")

         results["test_acc"].append(test_acc)
         results["test_loss"].append(test_loss)
         results["train_acc"].append(train_acc)
         results["train_loss"].append(train_loss)
         
    return results

def train_for_confmat (model:torch.nn.Module,
          train_dataloader:torch.utils.data.DataLoader,
          test_dataloadeer:torch.utils.data.DataLoader,
          loss_fn:torch.nn.Module,
          optimizer:torch.optim.Optimizer,
          epochs:int,
          device:torch.device):
    """Stejná jako train funkce akorát vrací i test i hodnoty potřebné pro confusional matrix"""
    def test_step_confmat(model:torch.nn.Module,
                  loss_fn:torch.nn.Module,
                  dataloader:torch.utils.data.DataLoader,
                  device:torch.device): #-> Tuple[float,float]):
        """Funkce stejná jako trainakorát vací navíc i list predikcí"""
        test_loss, test_acc =0,0
        model.eval()
        confmat_preds =[]
        
        with torch.inference_mode():
            for batch,(X,y) in enumerate(dataloader):
                X,y = X.to(device), y.to(device)

                y_test_pred = model(X)
                loss = loss_fn(y_test_pred,y)
                test_loss += loss.item()

                test_pred_labels = y_test_pred.argmax(dim=1)
                test_acc += (test_pred_labels == y).sum().item()/len(test_pred_labels)
                conf_pred = torch.softmax(y_test_pred.squeeze(),dim=0).argmax(dim=1)
                confmat_preds.append(conf_pred.cpu())  #? vytvoří číslo classy kterou model rozhodl jyko sperávnou
        test_loss = test_loss/len(dataloader)
        test_acc = test_acc/len(dataloader)
        
        return test_loss, test_acc, confmat_preds
    
    results = {"train_loss":[],
               "train_acc":[],
               "test_loss":[],
               "test_acc":[]}
    for epochs in tqdm(range(epochs)):
         train_loss, train_acc = train_step(model=model,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer,
                                            dataloader=train_dataloader,
                                            device=device)
         test_loss, test_acc, confmat_preds = test_step_confmat(model=model,
                                         loss_fn=loss_fn,
                                         dataloader=test_dataloadeer,
                                         device=device)
         
         print(f"Epocha: {epochs+1} | Train_loss: {train_loss} | Test_loss: {test_loss} | Train_acc: {train_acc} | Test_acc: {test_acc}")

         results["test_acc"].append(test_acc)
         results["test_loss"].append(test_loss)
         results["train_acc"].append(train_acc)
         results["train_loss"].append(train_loss)
         
    return results, confmat_preds