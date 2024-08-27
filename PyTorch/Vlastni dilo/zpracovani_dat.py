import ultralytics
import torch
import cv2
import torch
import torch.onnx
import onnxscript
print(cv2.__version__)


# Model

model = torch.hub.load("ultralytics/yolov5", "yolov5m")  # or yolov5n - yolov5x6, custom
model.eval()  # Set the model to evaluation mode

# Create a dummy input tensor
dummy_input = torch.randn(1, 3, 640, 640)  # Adjust dimensions according to your model's input size

# Trace the model
traced_model = torch.jit.trace(model, dummy_input)

# Export the traced model to ONNX format
torch.onnx.export(
    traced_model,                # model to be exported
    dummy_input,                 # an example input tensor
    r"C:\Users\Gamer\Desktop\111\Programování\CNN\PyTorch\yolov5m.onnx",  # file path
    export_params=True,          # store the trained parameter weights inside the model file
    opset_version=11,            # ONNX version
    do_constant_folding=True,    # optimization
    input_names=['images'],      # input names
    output_names=['output'],     # output names
    dynamic_axes={'images': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # dynamic axes
)
"""# Images
img = r"C:\\Users\\Gamer\\Downloads\\il_1080xN.4614246939_sotc.jpg"  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print()
results.show()

torch.onnx.export(model,  export_params=True, opset_version=11, do_constant_folding=True,
                  input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
                  """