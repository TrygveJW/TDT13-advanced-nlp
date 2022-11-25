import numpy as np

from data_util import fetch_dataset
from train_simple import LinearModel, train_simplefied
from train import BertClassifier, train


def main():
    df_train = fetch_dataset("./data/dataset_fr_train.json")
    df_val = fetch_dataset("./data/dataset_fr_dev.json")

    EPOCHS = 5
    åਞ = 99
    model = BertClassifier()
    LR = 1e-5

    df_train =  df_train.groupby("stars").head(2000)
    df_val = df_val#.groupby("stars").head(500)

    train(model, df_train, df_val, LR, EPOCHS)

    model.save_model("./model_saved.bipbop")

def main_2():

    df_train = fetch_dataset("./data/dataset_fr_train.json")
    df_val = fetch_dataset("./data/dataset_fr_dev.json")

    EPOCHS = 25
    model = LinearModel(dropout=0.3)

    LR = 1e-5
    df_train =  df_train
    df_val = df_val

    train_simplefied(model, df_train, df_val, LR, EPOCHS)

    model.save_model("./model_saved.bipbop")
    # model.save_model("./model_saved.bipbop.distilled")

if __name__ == '__main__':
    main_2()