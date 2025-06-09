import os
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import v2 as transforms
from torch import nn
import torch.nn.functional as F

LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCH = 20
IMG_SIZE = 64
NUM_CLASSES = 1
KERNEL_SIZE = 3
MODEL_NAME = 'simple_CNN'
TEST_FOLDER = r'./data/test_data'
image_extensions = ('.jpg', '.jpeg', '.png')

class SimpleCNN(nn.Module):
    def __init__(self, kernel_size=3, num_classes=10):
        super(SimpleCNN, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(1, 32, kernel_size, padding)
        self.conv2 = nn.Conv2d(32, 64, kernel_size, padding)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

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

    model = SimpleCNN(num_classes=10)
    model.load_state_dict(torch.load(
        r'E:\Code\Github\CS114.P21_project\digit_classification\simple_CNN\models\simple_CNN_model_11_weights.pth'))
    model.to(device)
    model.eval()

    images = []
    predictions = []

    for file in os.listdir(TEST_FOLDER):
        if file.lower().endswith(image_extensions):
            img_path = os.path.join(TEST_FOLDER, file)

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