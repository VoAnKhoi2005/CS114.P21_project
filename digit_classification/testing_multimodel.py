import os
import pandas as pd
import torch
from PIL import Image
from torchvision.models import EfficientNet, efficientnet_b0
from torchvision.transforms import v2 as transforms
from torch import nn
import torch.nn.functional as F

from digit_classification.digit_net_v1.main import AdvancedDigitNet_v1
from digit_classification.simple_CNN_v1.main import SimpleCNN_v1

IMG_SIZE = 64
NUM_CLASSES = 10
TEST_FOLDER = r'./data/test_data'
image_extensions = ('.jpg', '.jpeg', '.png')

def main():
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    models = []

    digit_net_model = AdvancedDigitNet_v1()
    digit_net_model.load_state_dict(torch.load(r'./digit_net_v1/models/digit_net_v1_model_28_weights.pth'))
    digit_net_model.to(device)
    digit_net_model.eval()
    models.append(digit_net_model)

    simple_cnn_model = SimpleCNN_v1()
    simple_cnn_model.load_state_dict(torch.load(r'./simple_CNN_v1/models/simple_CNN_model_16_weights.pth'))
    simple_cnn_model.to(device)
    simple_cnn_model.eval()
    models.append(simple_cnn_model)

    images = []
    predictions = []

    for file in os.listdir(TEST_FOLDER):
        if file.lower().endswith(image_extensions):
            img_path = os.path.join(TEST_FOLDER, file)

            try:
                image = Image.open(img_path).convert("L")
                image = transform(image).unsqueeze(0).to(device)

                ensemble_outputs = []
                with torch.no_grad():
                    for model in models:
                        output = model(image)
                        prob = F.softmax(output, dim=1)
                        ensemble_outputs.append(prob)

                avg_prob = torch.mean(torch.stack(ensemble_outputs), dim=0)
                confidence, prediction = torch.max(avg_prob, dim=1)
                images.append(file)
                predictions.append(prediction.item())

            except Exception as e:
                # print(f"Error processing {file}: {e}")
                continue

    df = pd.DataFrame({'image': images, 'prediction': predictions}).to_csv("ensemble_prediction.csv", index=False)
    return

if __name__ == "__main__":
    main()