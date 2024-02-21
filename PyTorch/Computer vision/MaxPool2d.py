import torch
from torch import nn


torch.manual_seed(42)
images = torch.randn(size=(33,3,64,64))
test_image = images[0]

# Create a sinlge conv2d layer
conv_layer = nn.Conv2d(in_channels=3, 
                       out_channels=10,
                       kernel_size=(3, 3),
                       stride=1,
                       padding=0)

# Pass the data through the convolutional layer 
conv_output = conv_layer(test_image.unsqueeze(0))
print(conv_output.shape)
"""
conv_layer = nn.Conv2d(in_channels=3,
                       out_channels=10,
                       kernel_size=3,
                       stride=1,
                       padding=0)

conv_output = conv_layer(test_image.unsqueeze(0))
"""
print(f"Test image original shape: {test_image.shape}")
print(f"Test image with unsqueezed dimension: {test_image.unsqueeze(0).shape}")


max_pool_layer = nn.MaxPool2d(kernel_size=2)

test_images_trough_conv = conv_layer(test_image.unsqueeze(dim=0))
print(f"Shape after going trough conv_layer: {test_images_trough_conv.shape}")

test_image_trough_conv_and_max_pool = max_pool_layer(test_images_trough_conv)
print(f" Shape after going tough conv.layer() and max_pool_layer(): {test_image_trough_conv_and_max_pool.shape}")

#? tensor s podobným tvarem jako nase vyuzívané fotky
torch.manual_seed(42)
random_tensor = torch.randn(size=(1,1,2,2))
print(f"\nRandom tensor:\n{random_tensor}")
print(f"Random tensor shape: {random_tensor.shape}")
#? maxpool
max_pool_layer = nn.MaxPool2d(kernel_size=2)
#? Forward
max_pool_tensor = max_pool_layer(random_tensor)
print(f"\nMaxpool tensor: {max_pool_tensor}")
print(f"Maxpool tensro shape: {max_pool_tensor.shape}")






