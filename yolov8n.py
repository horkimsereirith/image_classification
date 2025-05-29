from ultralytics import YOLO
from PIL import Image

model = YOLO("yolov8n.pt")  # Smallest model (faster), or use yolov8s.pt

results = model("imgs/cat_2.jpg")  # Replace with your image path

# Print detected classes
for result in results:
    for box in result.boxes:
        cls = int(box.cls[0])
        if model.names[cls] == "person":
            print("Person detected!")
        else:
            print("Detected:", model.names[cls])