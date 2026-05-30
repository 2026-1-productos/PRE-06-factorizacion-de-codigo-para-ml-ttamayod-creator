import pandas as pd
from sklearn.model_selection import train_test_split

URL = (
    "http://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)


def load_data():

    return pd.read_csv(URL, sep=";")


def split_features_target(df):

    y = df["quality"]
    x = df.copy()
    x.pop("quality")
    return x, y


def load_train_test():

    df = load_data()
    x, y = split_features_target(df)
    return train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=123456,
    )
