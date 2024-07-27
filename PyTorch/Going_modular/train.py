if __name__ == '__main__':      #! problem s ukladanim modelu -> koncoky modelu vyřešit!!!
    import sys
    import os
    import torch
    from torchvision import transforms
    from pathlib import Path
    from torch import nn

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #! umožňuje tahat data ze složky

    from Going_modular import model_builder,data_setup, engine, utils

    BATCH_SIZE = 32
    NUM_EPOCHS = 5
    HIDDEN_UNITIS = 10
    LR = 0.001

    data_path = Path("data/")
    image_path = data_path / "pizza_steak_sushi"

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
                    model_name="Model_goi_mod_00")