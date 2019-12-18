import numpy as np
import tiresias.core.mechanisms as dp

def test_laplace_noise():
    x = 100.0
    x_hat = dp.laplace_noise(x, sensitivity=1.0, epsilon=0.1)
    assert type(x_hat) == float

def test_count():
    x = np.random.normal(size=1000)
    x_hat = dp.count(x, epsilon=100.0)
    assert (x_hat - len(x))**2 < 1.0

def test_median():
    x = np.random.normal(size=1000)
    x_hat = dp.median(x, epsilon=100.0)
    assert (x_hat - np.median(x))**2 < 1.0

def test_mean():
    x = np.random.normal(size=1000)
    x_hat = dp.mean(x, epsilon=100.0)
    assert (x_hat - np.mean(x))**2 < 1.0

def test_sum():
    x = np.random.normal(size=1000)
    x_hat = dp.sum(x, epsilon=100.0)
    assert (x_hat - np.sum(x))**2 / len(x) < 1.0

def test_finite_categorical_str():
    x = "hello"
    domain = ["hello", "world"]
    x_hat = dp.finite_categorical(x, domain, epsilon=0.1)
    assert x_hat in domain

def test_finite_categorical_int():
    x = 1
    domain = [0, 1, 2, 3]
    x_hat = dp.finite_categorical(x, domain, epsilon=0.1)
    assert x_hat in domain

def test_bounded_continuous():
    x = 0.5
    x_hat = dp.bounded_continuous(x, 0.0, 1.0, epsilon=0.1)
    assert type(x_hat) == float
