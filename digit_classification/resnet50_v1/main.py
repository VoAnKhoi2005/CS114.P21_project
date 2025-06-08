import sys
from torch.utils.data import DataLoader
from torchvision import transforms
import digit_classification.engine.Dataset as dataset
from digit_classification.engine.Logger import Logger
from digit_classification.engine.load_data import load_from_csv
from digit_classification.engine.env_util import get_env

LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCH = 20
IMG_SIZE = 224

def main():
    sys.stdout = Logger("train.log")
    sys.stderr = sys.stdout

    data_folder = get_env('DATA_FOLDER')

    train_data, train_labels, val_data, val_labels = load_from_csv(data_folder)
    print(f"Number of train images: {len(train_data)}\nNumber of val images: {len(val_data)}\n")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = dataset.CustomIterableDataset(train_data, train_labels, transform)
    test_dataset = dataset.CustomIterableDataset(val_data, val_labels, transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return


if __name__ == "__main__":
    main()
