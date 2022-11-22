import numpy as np

from data_util import fetch_dataset, transform_tokenize
from train_simple import LinearModel, train_simplefied
from train import BertClassifier, train


def main():
    df_train = fetch_dataset("./data/dataset_fr_train.json")
    df_val = fetch_dataset("./data/dataset_fr_dev.json")

    # transform_tokenize(df_train)
    # exit()
    EPOCHS = 7
    åਞ = 99
    model = BertClassifier()
    LR = 1e-6
    print("read")
    # df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
    #                                      [int(.8 * len(df)), int(.9 * len(df))])
    df_train =  df_train#.groupby("stars").head(1000)
    df_val = df_val#.groupby("stars").head(1000)
    # print(df_train)
    # print(df_val)

    print("smp")
    train(model, df_train, df_val, LR, EPOCHS)

    model.save_model("./model_saved.bipbop")

def main_2():

    df_train = fetch_dataset("./data/dataset_fr_train.json")
    df_val = fetch_dataset("./data/dataset_fr_dev.json")

    EPOCHS = 50
    model = LinearModel()
    LR = 1e-6
    print("read")
    df_train =  df_train#.groupby("stars").head(500)
    df_val = df_val#.groupby("stars").head(500)

    print("smp")
    train_simplefied(model, df_train, df_val, LR, EPOCHS)

    model.save_model("./model_saved.bipbop")

if __name__ == '__main__':
    main_2()