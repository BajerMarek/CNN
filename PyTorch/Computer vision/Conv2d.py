import torch
from torch import nn


torch.manual_seed(42)
images = torch.randn(size=(33,3,64,64))
test_images = images[0]

print(f"Image batch shape: {images.shape}")
print(f"Single image shape: {test_images}")
print(f"Test image:\n {test_images}")

conv_layer = nn.Conv2d(in_channels=3,
                       out_channels=10,
                       kernel_size=3,
                       stride=1,
                       padding=0)

conv_output = conv_layer(test_images)
print("=============================")
print(conv_output)





