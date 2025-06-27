import os
import shutil
import pandas as pd
import torch
from PIL import Image
from pillow_heif import register_heif_opener
from torchvision.transforms import v2 as transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from digit_classification.simple_CNN_v2.main import SimpleCNN_v2
import math
from matplotlib.widgets import Button

register_heif_opener()

# Constants
IMG_SIZE = 64
TEST_FOLDER = r'E:\Code\Github\CS114.P21_project\digit_classification\data\image_raw_v1\sorted'
MODEL_PATH = r'models/simple_CNN_v2_model_38_weights.pth'
WRONG_FOLDER = "wrong_predictions"
image_extensions = ('.jpg', '.jpeg', '.png', '.jfif', '.heic')


class WrongPredictionViewer:
    def __init__(self, csv_path, batch_size=12, cols=4):
        self.df = pd.read_csv(csv_path)
        self.wrong = self.df[self.df['actual'] != self.df['predicted']].reset_index(drop=True)
        self.batch_size = batch_size
        self.cols = cols
        self.rows = math.ceil(batch_size / cols)
        self.current_index = 0

        self.fig, self.axes = plt.subplots(self.rows, self.cols, figsize=(self.cols * 2, self.rows * 2.2))
        self.axes = self.axes.flatten()

        # Add "Next" button
        ax_next = self.fig.add_axes([0.45, 0.01, 0.1, 0.05])  # [left, bottom, width, height]
        self.btn = Button(ax_next, "Next")
        self.btn.on_clicked(self.show_next_batch)

    def show_next_batch(self, event=None):
        start = self.current_index
        end = min(start + self.batch_size, len(self.wrong))
        batch = self.wrong.iloc[start:end]

        for ax in self.axes:
            ax.clear()
            ax.axis("off")

        for ax, (_, row) in zip(self.axes, batch.iterrows()):
            try:
                img = Image.open(row['image_path']).convert("L")
                ax.imshow(img, cmap='gray')
                fname = os.path.basename(row['image_path'])
                ax.set_title(f"A:{row['actual']} P:{row['predicted']}\n{fname}", fontsize=8)
                ax.axis("off")
            except Exception as e:
                ax.set_title("Error", fontsize=8)
                print(f"Error loading image {row['image_path']}: {e}")

        self.fig.canvas.draw()
        self.current_index += self.batch_size

        if self.current_index >= len(self.wrong):
            self.btn.label.set_text("Done")
            self.btn.on_clicked(None)  # Disable button

def main():
    csv_path = "evaluation_result.csv"
    viewer = WrongPredictionViewer(csv_path, batch_size=9, cols=3)
    viewer.show_next_batch()
    plt.show()
    return

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

    model = SimpleCNN_v2()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)
    model.eval()

    y_true, y_pred, image_paths = [], [], []

    for label_folder in sorted(os.listdir(TEST_FOLDER)):
        label_path = os.path.join(TEST_FOLDER, label_folder)
        if not os.path.isdir(label_path) or not label_folder.isdigit():
            continue

        actual_label = int(label_folder)

        for file in os.listdir(label_path):
            if not file.lower().endswith(image_extensions):
                continue

            img_path = os.path.join(label_path, file)
            try:
                image = Image.open(img_path).convert("L")
                image = transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(image)
                    predicted_class = torch.argmax(output, dim=1).item()

                y_true.append(actual_label)
                y_pred.append(predicted_class)
                image_paths.append(img_path)

                if predicted_class != actual_label:
                    save_path = os.path.join(WRONG_FOLDER, f"actual_{actual_label}_pred_{predicted_class}")
                    os.makedirs(save_path, exist_ok=True)
                    shutil.copy(img_path, os.path.join(save_path, file))

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    # Save predictions
    df = pd.DataFrame({
        'image_path': image_paths,
        'actual': y_true,
        'predicted': y_pred
    })
    df.to_csv("evaluation_result.csv", index=False)

    # Display and save confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    main()
