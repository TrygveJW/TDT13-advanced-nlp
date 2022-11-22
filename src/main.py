import numpy as np

from data_util import fetch_dataset, transform_tokenize
from train import BertClassifier, train


def main():
    df_train = fetch_dataset("./data/dataset_fr_train.json")
    df_val = fetch_dataset("./data/dataset_fr_dev.json")

    # transform_tokenize(df_train)
    # exit()
    EPOCHS = 5
    model = BertClassifier()
    LR = 1e-6
    print("read")
    # df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
    #                                      [int(.8 * len(df)), int(.9 * len(df))])
    df_train = df_train[:500]
    df_val = df_val[:500]

    print("smp")
    train(model, df_train, df_val, LR, EPOCHS)


if __name__ == '__main__':
    main()