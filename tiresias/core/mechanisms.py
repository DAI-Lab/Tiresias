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

def median(x, epsilon):
    """
    This function computes the differentially private estimate of the median 
    using the smooth sensitivity approach proposed in [1]. The input to this 
    function can be either a list or a vector.

    [1] http://www.cse.psu.edu/~ads22/pubs/NRS07/NRS07-full-draft-v1.pdf
    """
    x = np.array(x)
    x.sort()
    m = (len(x) + 1) // 2
    smooth_sensitivity = []
    for k in range(1, len(x)):
        local_sensitivity = max(x[m+t] - x[m+t-k-1] for t in range(min(k, len(x)-m)))
        smooth_sensitivity.append(np.exp(-k * epsilon) * local_sensitivity)
    smooth_sensitivity = max(smooth_sensitivity)
    return laplace_noise(np.median(x), sensitivity=smooth_sensitivity, epsilon=epsilon)

def sample_and_aggregate(x, func, epsilon, nb_partitions):
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
    return median(np.array(results), epsilon)

def mean(x, epsilon):
    """
    This function computes the differentially private estimate of the average 
    using the smooth sensitivity and sample and aggregate approach.
    """
    nb_partitions = int(np.sqrt(len(x)))
    return sample_and_aggregate(x, np.mean, epsilon, nb_partitions)

def sum(x, epsilon):
    """
    This function computes the differentially private estimate of the sum 
    using the smooth sensitivity and sample and aggregate approach.
    """
    nb_partitions = int(np.sqrt(len(x)))
    return nb_partitions * sample_and_aggregate(x, np.sum, epsilon, nb_partitions)

def finite_categorical(x, domain, epsilon):
    """
    This function applies randomized response to a categorical variable.
    """
    assert x in domain
    if epsilon == float("inf"):
        return x
    p = (np.exp(epsilon) - 1) / (len(domain) - 1 + np.exp(epsilon))
    if np.random.random() < p:
        return x
    return np.random.choice(domain)

def bounded_continuous(x, low, high, epsilon):
    """
    This function applies randomized response to a bounded continuous variable.
    """
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
