import numpy as np
from tiresias.core.regression import LinearRegression
from tiresias.core.classification import LogisticRegression, GaussianNB, TiresiasClassifier

def handle_integrated(task, data):
    rows = []
    for row in data:
        rows.extend(row)
    x = np.array([[row[var] for var in task["inputs"]] for row in rows])
    y = np.array([row[task["output"]] for row in rows])

    if task["model"] == "GaussianNB":
        clf = GaussianNB(epsilon=task["epsilon"])
        clf.fit(x, y)
        return clf

    elif task["model"] == "LogisticRegression":
        clf = LogisticRegression(epsilon=task["epsilon"])
        clf.fit(x, y)
        return clf

    elif task["model"] == "LinearRegression":
        clf = LinearRegression(epsilon=task["epsilon"])
        clf.fit(x, y)
        return clf

    elif task["model"] == "Classification":
        clf = TiresiasClassifier(epsilon=task["epsilon"])
        clf.fit(x, y)
        return clf

    else:
        raise ValueError(task["model"])
