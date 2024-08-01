if __name__ == '__main__':
    import sys
    import os
    from pathlib import Path
    import torch
    import torchvision
    from torchvision import transforms
    import torchinfo
    from torchvision import datasets, transforms
    from torchinfo import summary
    from tqdm.auto import tqdm
    #! ***************************** Získání modulů *****************************
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #! umožňuje tahat data ze složky
    from Going_modular import data_setup, get_data, engine, utils, predict
    target_dir = "C:\\Users\\Gamer\\Desktop\\111\\Programování\CNN\data\\"
    data_name = "pizza_steak_sushi"
    github_link="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
    get_data.get_data_github(target_dir=target_dir,
                            data_name=data_name,
                            github_link=github_link)
    data_path = Path(target_dir)
    image_path = data_path / data_name
    train_dir = image_path / "train"
    test_dir = image_path / "test"

    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    transform = weights.transforms()
    train_dataloader, test_dataloader, class_names =data_setup.Create_dataloader(train_dir=train_dir,
                                                                            test_dir=test_dir,
                                                                            batch_size=32,
                                                                            num_workers=os.cpu_count(),
                                                                            transform=transform)
    model = torchvision.models.efficientnet_b2(weights=weights).to(device)
    
    for pram in model.features.parameters():
        pram.requires_grad = False
    outpt_shape = len(class_names)
    model.classifier = torch.nn.Sequential(torch.nn.Dropout(p=0.2,inplace=True),
                                           torch.nn.Linear(in_features=1408,
                                                           out_features=outpt_shape,
                                                           bias=True,
                                                           device=device))
    summary(model=model,
            input_size=(32,3,224,224),
            col_names=["input_size","output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])
    

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                lr=0.001)
    torch.manual_seed(42)
    from timeit import default_timer as timer


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

    def train_for_confmat_2 (model:torch.nn.Module,
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
                    test_pred_probs = torch.softmax(y_test_pred.squeeze(),dim=0)

                    conf_pred = torch.softmax(y_test_pred.squeeze(),dim=0).argmax(dim=1)
                    confmat_preds.append(conf_pred.cpu())  #? vytvoří číslo classy kterou model rozhodl jyko sperávnou
            test_loss = test_loss/len(dataloader)
            test_acc = test_acc/len(dataloader)

            return test_loss, test_acc, confmat_preds,

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

        return results, confmat_preds,

    start_time = timer()
    vysledek, confmat_preds = train_for_confmat_2(model=model,
                            train_dataloader=train_dataloader,
                            test_dataloadeer=test_dataloader,
                            loss_fn=loss_fn,
                            optimizer=optimizer,
                            epochs=10,
                            device=device)

    end_time = timer()


    print(f"[INFO] Celkový čas: {end_time-start_time:.3f} [s]")
    utils.plot_loss_curves(results=vysledek)

   

    utils.save_model(model=model,
                     target_dir="models",
                     model_name="Transfer_model_01.pth")
    
    target_dir = "C:\\Users\\Gamer\\Desktop\\111\\Programování\CNN\data\\"
    data_path = Path(target_dir)
    model_name = "Transfer_model_01.pth"
    model_dir = f"C:\\Users\\Gamer\\Desktop\\111\\Programování\CNN\\models\\Transfer_model_01.pth"
    predict.zobraz_fotku(image_path=data_path/"04-pizza-dad.jpeg",
                     model=model,    
                     model_save_path=model_dir,
                     class_names=class_names,
                     device=device)
    #! confusional metrix           ******************************* dokončit ***********************************
    print(confmat_preds)

    from torchmetrics import ConfusionMatrix
    from mlxtend.plotting import plot_confusion_matrix
    import matplotlib.pyplot as plt


    test_data = datasets.ImageFolder(root=test_dir,
                                     transform=transform,
                                     target_transform=None)
    confmat_preds_tensor = torch.cat(confmat_preds)
    confmat = ConfusionMatrix(task="multiclass",num_classes=len(class_names))
    print(confmat_preds_tensor.shape)
    print(type(confmat_preds_tensor))
    print(test_data.targets)
    print(type(test_data.targets))
    confmat_tensor = confmat(preds=confmat_preds_tensor,
                         target=torch.tensor(test_data.targets))
    fig, ax = plot_confusion_matrix(conf_mat=confmat_tensor.numpy(),
                                    class_names=class_names,
                                    figsize=(10,7))
    plt.show()
