"""
The `tiresias.core.mechanisms` module provides low-level implementations of 
differential privacy mechanisms such as the Laplace mechanism, 
sample-and-aggregate, and the staircase mechanism.
"""
import numpy as np

def laplace_noise(x, sensitivity, epsilon):
    """
    This function returns a differentially private estimate of the value `x` 
    given the sensitivity and epsilon parameters.
    """
    return np.random.laplace(loc=x, scale=sensitivity/epsilon)

def count(x, epsilon):
    """
    This function computes the differentially private estimate of the count 
    using the Laplace mechanism.
    """
    return laplace_noise(len(x), sensitivity=1, epsilon=epsilon)

def median(x, epsilon, delta=None):
    """
    This function computes the differentially private estimate of the median 
    using the approach proposed in [1]. It uses the Laplace mechanism on page 
    10 and the smooth sensitivity of the median derivation found on page 12.
    
    The resulting value is (epsilon-delta) differentially private. By default,
    if not specified, delta is set to `1/(100*len(x))` so that there is a 99%
    chance that differential privacy is satisfied for any individual.

    [1] http://www.cse.psu.edu/~ads22/pubs/NRS07/NRS07-full-draft-v1.pdf
    """
    if not delta:
        delta = 1.0 / (100 * len(x))
    alpha = epsilon / 2.0
    beta = epsilon / (2.0 * np.log(2.0 / delta))
    
    x = np.array(x)
    x.sort()
    m = (len(x) + 1) // 2
    smooth_sensitivity = []
    for k in range(0, len(x)-m):
        local_sensitivity = max(x[m+t] - x[m+t-k-1] for t in range(0, k+1))
        smooth_sensitivity.append(np.exp(-k * beta) * local_sensitivity)
    smooth_sensitivity = max(smooth_sensitivity)
    
    return np.median(x) + smooth_sensitivity/alpha * np.random.laplace()

def median_gaussian(x, epsilon, delta=None):
    """
    This function computes the differentially private estimate of the median 
    using the approach proposed in [1]. It uses the Gaussian mechanism on page 
    10 and the smooth sensitivity of the median derivation found on page 12.

    The resulting value is (epsilon-delta) differentially private. By default,
    if not specified, delta is set to `1/(100*len(x))` so that there is a 99%
    chance that differential privacy is satisfied for any individual.

    [1] http://www.cse.psu.edu/~ads22/pubs/NRS07/NRS07-full-draft-v1.pdf
    """
    if not delta:
        delta = 1.0 / (100 * len(x))
    alpha = epsilon / (5.0 * np.sqrt(2.0 * np.log(2.0/delta)))
    beta = epsilon / (4.0 * (1.0 + np.log(2.0/delta)))
    
    x = np.array(x)
    x.sort()
    m = (len(x) + 1) // 2
    smooth_sensitivity = []
    for k in range(0, len(x)-m):
        local_sensitivity = max(x[m+t] - x[m+t-k-1] for t in range(0, k+1))
        smooth_sensitivity.append(np.exp(-k * beta) * local_sensitivity)
    smooth_sensitivity = max(smooth_sensitivity)

    return np.median(x) + smooth_sensitivity/alpha * np.random.normal()

def sample_and_aggregate(x, func, epsilon, nb_partitions, delta=None):
    """
    This function computes the differentially private estimate of a function 
    using the sample and aggregate approach in [1]. We obtain an estimate of
    the function value by computing it on samples of the data and then use 
    the differentially private median to aggregate the results.
    """
    results = []
    x = np.array(x)
    np.random.shuffle(x)
    for partition in np.array_split(x, nb_partitions):
        results.append(func(partition))
    return median(np.array(results), epsilon, delta=delta)

def mean(x, epsilon, delta=None):
    """
    This function computes the differentially private estimate of the average 
    using the smooth sensitivity and sample and aggregate approach.
    """
    nb_partitions = int(np.sqrt(len(x)))
    return sample_and_aggregate(x, np.mean, epsilon, nb_partitions, delta=delta)

def sum(x, epsilon, delta=None):
    """
    This function computes the differentially private estimate of the sum 
    using the smooth sensitivity and sample and aggregate approach.
    """
    nb_partitions = int(np.sqrt(len(x)))
    return nb_partitions * sample_and_aggregate(x, np.sum, epsilon, nb_partitions, delta=delta)

def finite_categorical(x, domain, epsilon, delta=None):
    """
    This function applies randomized response to a categorical variable. The
    input can be either a np.array or a single value. There is no restriction
    on the value type but in most scenarios, the value will be either an int 
    or a string.
    """
    assert len(set(domain)) == len(domain)

    if type(x) == np.ndarray:
        for _x in x:
            assert _x in domain
        p = (np.exp(epsilon) - 1) / (len(domain) - 1 + np.exp(epsilon))
        flags = np.random.random(size=x.shape) > p
        x[flags] = np.random.choice(list(domain), size=np.sum(flags))
        return x

    assert x in domain
    if epsilon == float("inf"):
        return x
    p = (np.exp(epsilon) - 1) / (len(domain) - 1 + np.exp(epsilon))
    if np.random.random() < p:
        return x
    return np.random.choice(list(domain))

def bounded_continuous(x, low, high, epsilon, delta=None):
    """
    This function applies randomized response to a bounded continuous variable. 
    The input `x` can be either a np.array or a single value.
    """
    if type(x) == np.ndarray:
        assert (x >= low).all() and (x <= high).all()
        if epsilon == float("inf"):
            return x
        return x + np.random.laplace(scale=(high-low)/epsilon, size=x.shape)

    assert x >= low and x <= high
    if epsilon == float("inf"):
        return x
    return x + np.random.laplace(scale=(high-low)/epsilon)

def staircase_mechanism(x, epsilon):
    """
    This function implements the staircase mechanism from [2] for local differential privacy.

    [2] https://arxiv.org/pdf/1212.1186.pdf
    """
    delta, gamma = 1.0, 0.5
    S = np.random.choice([-1.0, 1.0])
    G = np.random.geometric(np.exp(-epsilon))
    U = np.random.uniform()
    B = np.random.binomial(n=1, p=((1 - gamma)*np.exp(-epsilon)) / (gamma + (1 - gamma)*np.exp(-epsilon)))
    noise = S * ((1-B)*((G + gamma*U)*delta) + B * ((G + gamma + (1 - gamma) * U) * delta))
    return x + noise
