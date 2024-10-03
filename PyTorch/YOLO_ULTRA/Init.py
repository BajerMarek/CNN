from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.info()

results = model.train(data="coco8.yaml", epochs=10, imgsz=640)

# Run inference with the YOLOv8n model on the 'bus.jpg' image
results = model("data\Mulberry_Street_NYC.jpg")

from PIL import Image

from ultralytics import YOLO

# Load a pretrained YOLO11n model


# Run inference on 'bus.jpg'


# Visualize the results
for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array


    # Show results to screen (in supported environments)
    r.show()

    # Save results to disk
    r.save(filename=f"results{i}.jpg")