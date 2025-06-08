import os
from digit_classification.engine.load_data import load_image_path_to_csv

DATA_PATH = r'E:\Code\Github\CS114.P21_project\digit_classification\data\image_raw'

def main():
    os.makedirs(DATA_PATH, exist_ok=True)

    load_image_path_to_csv(DATA_PATH)
    return

if __name__ == '__main__':
    main()