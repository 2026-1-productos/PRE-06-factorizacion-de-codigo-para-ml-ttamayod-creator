from sklearn.neighbors import KNeighborsRegressor

from src.data import load_train_test
from src.metrics import report
from src.model import best_estimator, save_estimator

N_NEIGHBORS = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51]


def main():

    x_train, x_test, y_train, y_test = load_train_test()

    candidates = [KNeighborsRegressor(n_neighbors=k) for k in N_NEIGHBORS]

    estimator = best_estimator(candidates, x_train, x_test, y_train, y_test)

    report(estimator, x_train, x_test, y_train, y_test)

    save_estimator(estimator)


if __name__ == "__main__":
    main()
