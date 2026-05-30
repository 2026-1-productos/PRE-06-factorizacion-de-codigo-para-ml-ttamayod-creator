from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true, y_pred):

    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def print_metrics(title, metrics):

    print()
    print(f"{title}:")
    print(f"  MSE: {metrics['mse']}")
    print(f"  MAE: {metrics['mae']}")
    print(f"  R2: {metrics['r2']}")


def report(estimator, x_train, x_test, y_train, y_test):

    print()
    print(estimator, ":", sep="")

    train_metrics = compute_metrics(y_train, estimator.predict(x_train))
    print_metrics("Metricas de entrenamiento", train_metrics)

    test_metrics = compute_metrics(y_test, estimator.predict(x_test))
    print_metrics("Metricas de testing", test_metrics)
