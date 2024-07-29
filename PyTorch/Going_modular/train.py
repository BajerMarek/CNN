if __name__ == '__main__':      #! problem s ukladanim modelu -> koncoky modelu vyřešit!!!
    import sys
    import os
    import torch
    from torchvision import transforms
    from pathlib import Path
    from torch import nn
    """    
    import argparse
    praser = argparse.ArgumentParser(description="Tento program spusti PyTorch model",
                                     add_help="je potřeba zadat tyto argumenty:\nBATCH_SIZE\nNUM_EPOCHS\nHIDDEN_UNITIS\nLR")
    praser.add_argument("-BATCH_SIZE","--BATCH_SIZE",type=int,required=True,help="určí batchsize modelu")
    praser.add_argument("-NUM_EPOCHS","--NUM_EPOCHS",type=int,required=True,help="určí kolikrat se bude opakovat ucení modelu -> pocet opakováni v loopu ")
    praser.add_argument("-HIDDEN_UNITIS","--HIDDEN_UNITIS",type=int,required=True,help="určí s kolikati neurony bude síť pracovat -> bude je mít")
    praser.add_argument("-LR","--LR",type=int,required=True,help="určí s jakou rychlostí se bude model ucit v jednom opakování")"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #! umožňuje tahat data ze složky

    from Going_modular import model_builder,data_setup, engine, utils, get_data

    BATCH_SIZE = 32
    NUM_EPOCHS = 5
    HIDDEN_UNITIS = 10
    LR = 0.001

    target_dir = "data/"
    data_name = "pizza_steak_sushi"
    github_link="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"

    get_data.get_data_github(target_dir=target_dir,
                             data_name=data_name,
                             github_link=github_link)
    
    data_path = Path(target_dir)
    image_path = data_path / data_name

    train_dir = image_path / "train"
    test_dir = image_path / "test"

    transform = transforms.Compose([transforms.Resize(size=(64,64)),
                                    transforms.TrivialAugmentWide(num_magnitude_bins=31),
                                    transforms.ToTensor()])

    train_dataloader, test_dataloader, class_names = data_setup.Create_dataloader(train_dir=train_dir,
                                                                                test_dir=test_dir,
                                                                                transform=transform,
                                                                                batch_size=BATCH_SIZE,
                                                                                num_workers=os.cpu_count())  #? doplnit

    model = model_builder.TinyVgg(input_shape=3,
                                hidden_units=HIDDEN_UNITIS,
                                output_shape=len(class_names))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                lr=LR)

    vysledk = engine.train(model=model,
                        train_dataloader=train_dataloader,
                        test_dataloadeer=test_dataloader,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        epochs=NUM_EPOCHS,
                        device= device)        #? doplnit

    utils.save_model(model=model,
                    target_dir="models",
                    model_name="Model_goi_mod_00.pth")