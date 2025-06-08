import os
import random
import re
import pandas as pd
from git import Repo

def clone_github():
    # Read GitHub link from a text file
    with open("text.txt", "r", encoding="utf-8") as file:
        text = file.read()

    pattern = r"https?://github\.com/[^\s/]+/[^\s/]+"
    github_links = re.findall(pattern, text)

    print(len(github_links))

    # clone GitHub
    destination_path = r'E:\Code\number_classification\data\raw_number_data'
    index = 0

    for link in github_links:
        path = os.path.join(destination_path, f"repo_{index}")
        index += 1
        os.makedirs(path, exist_ok=True)

        try:
            Repo.clone_from(link, path)
            print(f"Cloned {link} into {path}")
        except Exception as e:
            print(f"Failed to clone {link} into {path}. Error: {e}")

def load_image_path_to_csv(data_folder, split: float = 0.8):
    img_paths = []
    labels = []
    image_extensions = {'.jpg', '.jpeg', '.png'}

    for folder in os.listdir(data_folder):
        folder_path = os.path.join(data_folder, folder)

        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):
            if os.path.isfile(os.path.join(folder_path, file)) and os.path.splitext(file)[1].lower() in image_extensions:
                img_path = os.path.join(folder_path, file)
                img_paths.append(img_path)
                labels.append(folder)
            else:
                print(os.path.join(folder_path, file))

    print(f"Number of images: {len(img_paths)}")

    combined = list(zip(img_paths, labels))
    if not combined:
        print("No valid images found. Check folder paths and file extensions.")
        return

    random.shuffle(combined)
    img_paths, labels = zip(*combined)
    img_paths = list(img_paths)
    labels = list(labels)

    split_index = int(len(img_paths) * split)
    train_dataset_path = img_paths[:split_index]
    train_label = labels[:split_index]
    val_dataset_path = img_paths[split_index:]
    val_label = labels[split_index:]

    train = pd.DataFrame({
        "path": train_dataset_path,
        "label": train_label
    })
    train.to_csv("train_data.csv", index=False)

    val = pd.DataFrame({
        "path": val_dataset_path,
        "label": val_label
    })
    val.to_csv("val_data.csv", index=False)

def load_from_csv(data_folder):
    train = pd.read_csv(os.path.join(data_folder, "train_data.csv"))
    train_data = train['path'].values
    train_label = train['label'].astype(int).values

    test = pd.read_csv(os.path.join(data_folder, "val_data.csv"))
    test_data = test['path'].values
    test_label = test['label'].astype(int).values
    return train_data, train_label, test_data, test_label