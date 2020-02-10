import time
import pandas as pd

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR

from tiresias.core import classification as classification
from tiresias.core import regression as regression
from tiresias.benchmark.helpers import make_ldp, FederatedLearningClassifier, FederatedLearningRegressor

def benchmark(X, y, epsilon, delta, problem_type):
    """
    This function takes in a standard tabular dataset (X, y) and a problem 
    problem_type (i.e. classification or regression) and evaluates a suite of
    machine learning models and differential privacy mechanisms on it.

    Note that this is *not* deterministic. You must set a random seed for
    numpy and pytorch before calling this function.
    """
    scalar = RobustScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)
    if problem_type == "regression":
        scalar = StandardScaler()
        y_train = scalar.fit_transform(y_train.reshape(-1,1))[:,0]
        y_test = scalar.transform(y_test.reshape(-1,1))[:,0]
    return {
        "classification": benchmark_classification,
        "regression": benchmark_regression,
    }[problem_type](X_train, X_test, y_train, y_test, epsilon, delta)

def benchmark_classification(X_train, X_test, y_train, y_test, epsilon, delta):
    report = []

    # LogisticRegression - Local Differential Privacy
    for C in [1.0, 10.0, 100.0]:
        model = LogisticRegression(C=C, solver='lbfgs', multi_class='auto')
        start = time.time()
        X_train_ldp, y_train_ldp = make_ldp(X_train, y_train, epsilon, delta)
        model.fit(X_train_ldp, y_train_ldp)
        report.append({
            "type": "bounded",
            "model": type(model).__name__,
            "hyperparameters": "C=%s" % C,
            "epsilon": epsilon,
            "accuracy": model.score(X_test, y_test),
            "time": time.time() - start
        })

        model = LinearSVC(C=C, max_iter=10000)
        start = time.time()
        X_train_ldp, y_train_ldp = make_ldp(X_train, y_train, epsilon, delta)
        model.fit(X_train_ldp, y_train_ldp)
        report.append({
            "type": "bounded",
            "model": type(model).__name__,
            "hyperparameters": "C=%s" % C,
            "epsilon": epsilon,
            "accuracy": model.score(X_test, y_test),
            "time": time.time() - start
        })

    # RandomForestClassifier - Local Differential Privacy
    for n_estimators in [10, 50, 100]:
        model = RandomForestClassifier(n_estimators=n_estimators)
        start = time.time()
        X_train_ldp, y_train_ldp = make_ldp(X_train, y_train, epsilon, delta)
        model.fit(X_train_ldp, y_train_ldp)
        report.append({
            "type": "bounded",
            "model": type(model).__name__,
            "hyperparameters": "n_estimators=%s" % n_estimators,
            "epsilon": epsilon,
            "accuracy": model.score(X_test, y_test),
            "time": time.time() - start
        })

    # LogisticRegression - Integrated
    for C in [1.0, 10.0, 100.0]:
        model = classification.LogisticRegression(epsilon=epsilon, C=C)
        start = time.time()
        model.fit(X_train, y_train)
        report.append({
            "type": "integrated",
            "model": type(model).__name__,
            "hyperparameters": "C=%s" % C,
            "epsilon": epsilon,
            "accuracy": model.score(X_test, y_test),
            "time": time.time() - start
        })

    # NaiveBayes - Integrated
    model = classification.GaussianNB(epsilon=epsilon)
    start = time.time()
    model.fit(X_train, y_train)
    report.append({
        "type": "integrated",
        "model": type(model).__name__,
        "hyperparameters": "",
        "epsilon": epsilon,
        "accuracy": model.score(X_test, y_test),
        "time": time.time() - start
    })

    # FederatedLearningClassifier - Gradient
    for epochs in [8, 16, 32]:
        model = FederatedLearningClassifier(epsilon, delta, epochs=epochs, lr=1e-2)
        start = time.time()
        model.fit(X_train, y_train)
        report.append({
            "type": "gradient",
            "model": type(model).__name__,
            "hyperparameters": "epochs=%s" % epochs,
            "epsilon": epsilon,
            "accuracy": model.score(X_test, y_test),
            "time": time.time() - start
        })

    return pd.DataFrame(report)

def benchmark_regression(X_train, X_test, y_train, y_test, epsilon, delta):
    report = []

    # SGDRegressor - Local Differential Privacy
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        model = SGDRegressor(alpha=alpha, loss='huber', max_iter=1000, tol=1e-3)
        start = time.time()
        X_train_ldp, y_train_ldp = make_ldp(X_train, y_train, epsilon, delta, classification=False)
        model.fit(X_train_ldp, y_train_ldp)
        report.append({
            "type": "bounded",
            "model": type(model).__name__,
            "hyperparameters": "alpha=%s" % alpha,
            "epsilon": epsilon,
            "accuracy": model.score(X_test, y_test),
            "time": time.time() - start
        })

    # LinearSVR - Local Differential Privacy
    for C in [1.0, 10.0, 100.0, 1000.0]:
        model = LinearSVR(C=C, max_iter=10000)
        start = time.time()
        X_train_ldp, y_train_ldp = make_ldp(X_train, y_train, epsilon, delta, classification=False)
        model.fit(X_train_ldp, y_train_ldp)
        report.append({
            "type": "bounded",
            "model": type(model).__name__,
            "hyperparameters": "C=%s" % C,
            "epsilon": epsilon,
            "accuracy": model.score(X_test, y_test),
            "time": time.time() - start
        })

    # RandomForestRegressor - Local Differential Privacy
    for n_estimators in [10, 50, 100, 1000]:
        model = RandomForestRegressor(n_estimators=n_estimators)
        start = time.time()
        X_train_ldp, y_train_ldp = make_ldp(X_train, y_train, epsilon, delta)
        model.fit(X_train_ldp, y_train_ldp)
        report.append({
            "type": "bounded",
            "model": type(model).__name__,
            "hyperparameters": "n_estimators=%s" % n_estimators,
            "epsilon": epsilon,
            "accuracy": model.score(X_test, y_test),
            "time": time.time() - start
        })

    # LinearRegression - Integrated
    model = regression.LinearRegression(epsilon=epsilon)
    start = time.time()
    model.fit(X_train, y_train)
    report.append({
        "type": "integrated",
        "model": type(model).__name__,
        "hyperparameters": "",
        "epsilon": epsilon,
        "accuracy": model.score(X_test, y_test),
        "time": time.time() - start
    })

    # FederatedLearningRegressor - Gradient
    for epochs in [8, 16, 32]:
        model = FederatedLearningRegressor(epsilon, delta, epochs=epochs, lr=1e-2)
        start = time.time()
        model.fit(X_train, y_train)
        report.append({
            "type": "gradient",
            "model": type(model).__name__,
            "hyperparameters": "epochs=%s" % epochs,
            "epsilon": epsilon,
            "accuracy": model.score(X_test, y_test),
            "time": time.time() - start
        })

    return pd.DataFrame(report)
