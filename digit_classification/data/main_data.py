import os
from digit_classification.engine.load_data import load_image_path_to_csv

DATA_PATH = r'E:\Code\Github\CS114.P21_project\digit_classification\data\image_raw'
DATA_PATH_V2 = r'E:\Code\data\digit\dataset'

def main():
    # os.makedirs(DATA_PATH, exist_ok=True)
    # #access image folder and load image path to train and val file
    # load_image_path_to_csv(DATA_PATH)

    load_image_path_to_csv(DATA_PATH_V2, split=0.9, version=2)
    return

if __name__ == '__main__':
    main()