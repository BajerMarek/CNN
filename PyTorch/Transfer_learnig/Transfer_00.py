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
    manual_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize(size=(224,224)),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
    transform_weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT        #? DEFAULT -> nejlepší možný(podlevýsledků na imagenetu)
    auto_transform = transform_weights.transforms()
    train_dataloader, test_dataloader, class_names =data_setup.Create_dataloader(train_dir=train_dir,
                                                                                test_dir=test_dir,
                                                                                batch_size=32,
                                                                                num_workers=os.cpu_count(),
                                                                                transform=auto_transform)
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)
    for pram in model.features.parameters():
        pram.requires_grad = False
                #? pravděpodobnost se ktero rozpojí náhodné nourony
    torch.manual_seed(42)
    output_shape = len(class_names)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2,inplace=True),
        torch.nn.Linear(in_features=1280,
                        out_features=output_shape,
                        bias=True)).to(device)
    summary(model=model,
            input_size=(32,3,224,224),
            col_names=["input_size","output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])      #? [BATCH_SIZE, COLOR_CHANELES, HEIGHT, WIDTH]
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                lr=0.001)

    from timeit import default_timer as timer
    start_time = timer()
    vysledek, confmat_preds = engine.train_for_confmat(model=model,
                            train_dataloader=train_dataloader,
                            test_dataloadeer=test_dataloader,
                            loss_fn=loss_fn,
                            optimizer=optimizer,
                            epochs=10,
                            device=device)
    end_time = timer()
    print(f"[INFO] Celkový čas: {end_time-start_time:.3f} [s]")
    utils.plot_loss_curves(results=vysledek)

    import random
    pocet_random_fotek = 1
    random_test_image_list = list(Path(test_dir).glob("*/*.jpg")) #? zíká všechny cetsy fotek do listu
    random_test_images = random.sample(population=random_test_image_list,
                                       k=pocet_random_fotek)
    iamge_path = "data\\04-pizza-dad.jpeg"
    for image_path in random_test_images:
        predict.pred_zobraz_fotku_PIL(model=model,
                                      class_names=class_names,
                                      image_path=image_path,
                                      image_size=(224,224),
                                      transform=auto_transform,
                                      device=device)

    utils.save_model(model=model,
                     target_dir="models",
                     model_name="Transfer_model_00.pth")
    
    target_dir = "C:\\Users\\Gamer\\Desktop\\111\\Programování\CNN\data\\"
    data_path = Path(target_dir)
    model_name = "Transfer_model_00.pth"
    model_dir = f"C:\\Users\\Gamer\\Desktop\\111\\Programování\CNN\\models\\Transfer_model_00.pth"
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
                                     transform=auto_transform,
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
