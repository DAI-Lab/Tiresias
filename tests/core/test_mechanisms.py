import tiresias.core.mechanisms as dp

def test_laplace_noise():
    x = 100.0
    x_hat = dp.laplace_noise(x, sensitivity=1.0, epsilon=0.1)
    assert type(x_hat) == float

def test_count():
    x = [0.0] * 100
    x_hat = dp.count(x, epsilon=0.1)
    assert type(x_hat) == float

def test_median():
    x = [0.0] * 100
    x_hat = dp.median(x, epsilon=0.1)
    assert type(x_hat) == float

def test_mean():
    x = [0.0] * 100
    x_hat = dp.mean(x, epsilon=0.1)
    assert type(x_hat) == float

def test_sum():
    x = [0.0] * 100
    x_hat = dp.sum(x, epsilon=0.1)
    assert type(x_hat) == float

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
