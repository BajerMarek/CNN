import sys
import os
from pathlib import Path
import torch
import torchvision
from torchvision import transforms
import torchinfo
from torchinfo import summary

device = "cuda" if torch.cuda.is_available() else "cpu"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #! umožňuje tahat data ze složky
from Going_modular import data_setup, get_data
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