import os
import pickle

from src.metrics import compute_metrics

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "estimator.pkl")


def best_estimator(candidates, x_train, x_test, y_train, y_test):

    best = None
    best_r2 = float("-inf")

    for estimator in candidates:
        estimator.fit(x_train, y_train)
        r2 = compute_metrics(y_test, estimator.predict(x_test))["r2"]
        if r2 > best_r2:
            best_r2 = r2
            best = estimator

    return best


def save_estimator(estimator):

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as file:
        pickle.dump(estimator, file)


def load_estimator():

    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)
