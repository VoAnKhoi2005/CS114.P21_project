import os
import sys
import time

import matplotlib
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms

import digit_classification.engine.Dataset as ds
from digit_classification.engine import History
from digit_classification.engine import train_loop, val_loop, plot_history
from digit_classification.engine.Logger import Logger
from digit_classification.engine.env_util import get_env
from digit_classification.engine.load_data import load_from_csv

matplotlib.use('TkAgg')

LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCH = 30
IMG_SIZE = 64
NUM_CLASSES = 1
KERNEL_SIZE = 5
MODEL_NAME = 'digit_net_v1'

class AdvancedDigitNet_v1(nn.Module):
    def __init__(self):
        super(AdvancedDigitNet_v1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def main():
    sys.stdout = Logger("train.log")
    sys.stderr = sys.stdout

    #Need .env file
    data_folder = get_env('DATA_FOLDER')

    train_data, train_labels, val_data, val_labels = load_from_csv(data_folder)
    print(f"Number of train images: {len(train_data)}\nNumber of val images: {len(val_data)}\n")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = ds.CustomIterableDataset(train_data, train_labels, grayscale=True, transform=transform)
    test_dataset = ds.CustomIterableDataset(val_data, val_labels, grayscale=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4)

    # show_batch(test_loader, IMG_SIZE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    model = AdvancedDigitNet_v1().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

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