import os
import sys
import time

import matplotlib
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms

import cv2
import numpy as np
from PIL import Image

import digit_classification.engine.Dataset as ds
from digit_classification.engine import History, show_batch
from digit_classification.engine import train_loop, val_loop, plot_history
from digit_classification.engine.Logger import Logger
from digit_classification.engine.env_util import get_env
from digit_classification.engine.load_data import load_from_csv

matplotlib.use('TkAgg')

LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCH = 35
IMG_SIZE = 64
NUM_CLASSES = 10
KERNEL_SIZE = 3
MODEL_NAME = 'simple_CNN_v3'

class SimpleCNN_v3(nn.Module):
    def __init__(self, kernel_size=3, num_classes=10):
        super(SimpleCNN_v3, self).__init__()
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

class RemoveLines:
    def __init__(self, line_spacing=30, thickness=1, inpaint_radius=1):
        self.line_spacing = line_spacing  # distance between lines
        self.thickness = thickness        # line thickness
        self.inpaint_radius = inpaint_radius

    def __call__(self, img):
        # Convert to grayscale
        img_np = np.array(img.convert("L"))

        # Step 1: Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(img_np)

        # Step 2: Binarize (invert for white background)
        _, binary = cv2.threshold(contrast, 127, 255, cv2.THRESH_BINARY_INV)

        # Step 3: Detect horizontal lines (lined paper)
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (img_np.shape[1] // 2, self.thickness))
        mask_h = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h)

        # Optional: Clean short artifacts
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_h)
        clean_mask = np.zeros_like(mask_h)
        for i in range(1, num_labels):
            width = stats[i, cv2.CC_STAT_WIDTH]
            if width > img_np.shape[1] // 3:  # keep only wide lines
                clean_mask[labels == i] = 255

        # Step 4: Inpaint
        result = cv2.inpaint(img_np, clean_mask, self.inpaint_radius, cv2.INPAINT_NS)

        final_binary = cv2.bitwise_not(result)

        return Image.fromarray(final_binary)


def main():
    sys.stdout = Logger("train.log")

    print("=" * 40)
    print("Training Configuration")
    print("=" * 40)
    print(f"MODEL_NAME: {MODEL_NAME}")
    print(f"IMG_SIZE: {IMG_SIZE}")
    print(f"NUM_CLASSES: {NUM_CLASSES}")
    print(f"LEARNING_RATE: {LEARNING_RATE}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"EPOCH: {EPOCH}")
    print(f"KERNEL_SIZE: {KERNEL_SIZE}")
    print("=" * 40)

    #Need .env file
    data_folder = get_env('DATA_FOLDER')

    train_data, train_labels, val_data, val_labels = load_from_csv(data_folder, data_version=3)
    print(f"Number of train images: {len(train_data)}\nNumber of val images: {len(val_data)}\n")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        RemoveLines(thickness=1, inpaint_radius=5),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = ds.CustomIterableDataset(train_data, train_labels, grayscale=True, transform=transform)
    test_dataset = ds.CustomIterableDataset(val_data, val_labels, grayscale=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4)

    # show_batch(test_loader, IMG_SIZE)
    # return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    model = SimpleCNN_v3(kernel_size=KERNEL_SIZE, num_classes=NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    os.makedirs(r'./models', exist_ok=True)
    history = History()
    for e in range(EPOCH):
        print(f"Epoch {e + 1}\n-------------------------------")

        if os.path.exists(rf'./models/{MODEL_NAME}_model_{e}_weights.pth'):
            model.load_state_dict(torch.load(f'./models/{MODEL_NAME}_model_{e}_weights.pth'))
            print(f"Skip training. Model weights loaded from {MODEL_NAME}_model_{e}_weights.pth")
            continue

        loss, acc = train_loop(train_loader, model, criterion, optimizer, device)
        history.train_loss.append(loss)
        history.train_accuracy.append(acc)

        loss, acc = val_loop(test_loader, model, criterion, device)
        history.test_loss.append(loss)
        history.test_accuracy.append(acc)

        torch.save(model.state_dict(), rf'./models/{MODEL_NAME}_model_{e}_weights.pth')
    print("Done!")

    plot_history(history)
    return


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_sec = end_time - start_time
    elapsed_min = elapsed_sec / 60
    print(f"Total runtime: {elapsed_min:.2f} minutes")