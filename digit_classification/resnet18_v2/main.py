import os
import sys
import time

import matplotlib
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import v2 as transforms

import digit_classification.engine.Dataset as ds
from digit_classification.engine import History, show_batch
from digit_classification.engine import train_loop, val_loop, plot_history
from digit_classification.engine.Logger import Logger
from digit_classification.engine.env_util import get_env
from digit_classification.engine.load_data import load_from_csv

matplotlib.use('TkAgg')

LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCH = 20
IMG_SIZE = 224
NUM_CLASSES = 10
MODEL_NAME = 'resnet18'

def main():
    sys.stdout = Logger("train.log")
    sys.stderr = sys.stdout

    #Need .env file
    data_folder = get_env('DATA_FOLDER')

    train_data, train_labels, val_data, val_labels = load_from_csv(data_folder)
    print(f"Number of train images: {len(train_data)}\nNumber of val images: {len(val_data)}\n")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ds.CustomIterableDataset(train_data, train_labels, transform=transform)
    test_dataset = ds.CustomIterableDataset(val_data, val_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4)

    # show_batch(train_loader, IMG_SIZE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {"params": model.fc.parameters(), "lr": 1e-3},
        {"params": model.layer4.parameters(), "lr": 1e-4},
    ], lr=1e-4)

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
