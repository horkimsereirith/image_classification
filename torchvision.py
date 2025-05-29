import torch
from torchvision import models, transforms
from PIL import Image
import requests

# Load labels
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = requests.get(LABELS_URL).text.strip().split("\n")

# Load the pre-trained model
model = models.resnet18(pretrained=True)
model.eval()

# Preprocess the image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def classify_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        _, index = torch.max(output, 1)
        predicted_label = labels[index]

        print(f"Predicted: {predicted_label}")

        if "person" in predicted_label.lower():
            print("This is a PERSON.")
        elif any(animal in predicted_label.lower() for animal in ["dog", "cat", "bird", "horse", "elephant", "zebra", "lion", "tiger", "bear"]):
            print("This is an ANIMAL.")
        else:
            print("This is something else.")

# Example usage
# Replace 'your_image.jpg' with your actual image file path
# classify_image("imgs/cat.jpg")
classify_image("imgs/cat_2.jpg")