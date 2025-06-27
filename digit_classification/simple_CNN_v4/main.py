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
MODEL_NAME = 'simple_CNN_v4'

class SimpleCNN_v4(nn.Module):
    def __init__(self, kernel_size=3, num_classes=10):
        super(SimpleCNN_v4, self).__init__()
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

class RemoveLines_v1:
    def __init__(
        self,
        canny_thresh1=50,
        canny_thresh2=150,
        min_line_length=100,
        max_line_gap=5,
        inpaint_radius=2,
        adapt_blocksize=15,
        adapt_C=4,
        cleanup_kernel_size=2,
        mask_dilate_iter=2
    ):
        self.c_thresh1        = canny_thresh1
        self.c_thresh2        = canny_thresh2
        self.min_line_len     = min_line_length
        self.max_line_gap     = max_line_gap
        self.inpaint_rad      = inpaint_radius
        self.adapt_block      = adapt_blocksize
        self.adapt_C          = adapt_C
        self.cleanup_kernel   = cleanup_kernel_size
        self.mask_dilate_iter = mask_dilate_iter

    def __call__(self, img):
        # 1) Grayscale & CLAHE
        gray = np.array(img.convert("L"))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 2) Edge → Hough → line‐mask
        edges = cv2.Canny(enhanced, self.c_thresh1, self.c_thresh2)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, threshold=80,
            minLineLength=self.min_line_len,
            maxLineGap=self.max_line_gap
        )
        mask = np.zeros_like(gray)
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                cv2.line(mask, (x1, y1), (x2, y2), 255, 1)
        # dilate mask a bit more aggressively
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=self.mask_dilate_iter)

        # 3) Inpaint to soften the line remnants
        color = np.array(img.convert("RGB"))
        inpainted = cv2.inpaint(color, mask, self.inpaint_rad, cv2.INPAINT_TELEA)
        cleaned = cv2.cvtColor(inpainted, cv2.COLOR_RGB2GRAY)

        # 4) Adaptive Gaussian threshold to binary
        binary = cv2.adaptiveThreshold(
            cleaned, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.adapt_block,
            self.adapt_C
        )

        # 5) Force‐white any pixel in our dilated line‐mask
        binary[mask > 0] = 255

        # 6) Tiny morphological opening to remove thin streaks
        small_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.cleanup_kernel, self.cleanup_kernel)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, small_kernel)

        return Image.fromarray(binary)



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
        RemoveLines_v1(),
        transforms.RandomRotation(10),
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

    model = SimpleCNN_v4(kernel_size=KERNEL_SIZE, num_classes=NUM_CLASSES).to(device)

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