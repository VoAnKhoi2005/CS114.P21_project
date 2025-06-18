import pandas as pd


def main():
    pred = pd.read_csv("prediction.csv")
    ensemble_pred = pd.read_csv("ensemble_prediction.csv")
    return

if __name__ == "__main__":
    main()