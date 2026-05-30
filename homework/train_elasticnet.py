from sklearn.linear_model import ElasticNet

from src.data import load_train_test
from src.metrics import report
from src.model import best_estimator, save_estimator

HYPERPARAMS = [
    (0.5, 0.5),
    (0.2, 0.2),
    (0.1, 0.1),
    (0.1, 0.05),
    (0.3, 0.2),
]


def main():

    x_train, x_test, y_train, y_test = load_train_test()

    candidates = [
        ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=12345)
        for alpha, l1_ratio in HYPERPARAMS
    ]

    estimator = best_estimator(candidates, x_train, x_test, y_train, y_test)

    report(estimator, x_train, x_test, y_train, y_test)

    save_estimator(estimator)


if __name__ == "__main__":
    main()
