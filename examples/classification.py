from sklearn.datasets import load_breast_cancer
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tiresias.core.classification import LogisticRegression as DPLogisticRegression

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

print(f1_score(y_test, LogisticRegression(C=100.0).fit(X_train, y_train).predict(X_test)))
print(f1_score(y_test, DPLogisticRegression(epsilon=2.0, C=100.0).fit(X_train, y_train).predict(X_test)))
print(f1_score(y_test, DPLogisticRegression(epsilon=4.0, C=100.0).fit(X_train, y_train).predict(X_test)))
print(f1_score(y_test, DPLogisticRegression(epsilon=8.0, C=100.0).fit(X_train, y_train).predict(X_test)))
print(f1_score(y_test, DPLogisticRegression(epsilon=16.0, C=100.0).fit(X_train, y_train).predict(X_test)))
print(f1_score(y_test, DPLogisticRegression(epsilon=32.0, C=100.0).fit(X_train, y_train).predict(X_test)))
print(f1_score(y_test, DPLogisticRegression(epsilon=64.0, C=100.0).fit(X_train, y_train).predict(X_test)))
