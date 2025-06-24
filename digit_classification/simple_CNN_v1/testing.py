import os
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import v2 as transforms
from torch import nn
import torch.nn.functional as F
import re

from digit_classification.digit_net_v1.main import AdvancedDigitNet_v1
from digit_classification.simple_CNN_v1.main import SimpleCNN_v1

LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCH = 20
IMG_SIZE = 64
NUM_CLASSES = 1
KERNEL_SIZE = 3
MODEL_NAME = 'simple_CNN_v1'
TEST_FOLDER = r'E:\Code\Github\CS114.P21_project\digit_classification\data\test_data'
image_extensions = ('.jpg', '.jpeg', '.png')

def main():
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    model = SimpleCNN_v1()
    model.load_state_dict(torch.load(r'models/simple_CNN_model_16_weights.pth'))
    model.to(device)
    model.eval()

    images = []
    predictions = []

    pattern = re.compile(r"^[a-f0-9]{32}  -\.(jpg|jpeg|png|bmp|gif)$", re.IGNORECASE)
    for file in os.listdir(TEST_FOLDER):
        if file.lower().endswith(image_extensions):
            img_path = os.path.join(TEST_FOLDER, file)

            if not pattern.match(file):
                continue

            try:
                image = Image.open(img_path).convert("L")
                image = transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(image)
                    predicted_class = torch.argmax(output, dim=1).item()

                images.append(file)
                predictions.append(predicted_class)

            except Exception as e:
                # print(f"Error processing {file}: {e}")
                continue

    df = pd.DataFrame({'image': images, 'prediction': predictions}).to_csv("prediction.csv", index=False)
    return

if __name__ == "__main__":
    main()