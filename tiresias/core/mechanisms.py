import numpy as np

def count(x, epsilon, delta):
    """
    This function computes the differentially private estimate of the count 
    using the Laplace mechanism.
    """
    return laplace_noise(len(x), sensitivity=1, epsilon=epsilon)

def laplace_noise(x, sensitivity, epsilon):
    """
    This function returns a differentially private estimate of the value `x` 
    given the sensitivity and epsilon parameters.
    """
    return np.random.laplace(loc=x, scale=sensitivity/epsilon)

def approx_bounds(x, epsilon=1.0, scale=0.1, base=2, bins=128, p=0.99999):
    """
    This function estimates the upper and lower bounds of the data using a 
    modification of the histogram approach from [1].

    [1] https://github.com/google/differential-privacy/
    """
    threshold = -np.log(2 - 2 * p) / epsilon
    cutoffs = scale * np.power(base, np.arange(0, bins//2))
    cutoffs = (-cutoffs[::-1]).tolist() + [0] + cutoffs.tolist()

    histogram = np.random.laplace(size=bins, scale=1.0 / epsilon)
    x = sorted(x.tolist())
    for i in range(len(cutoffs)-1, 0, -1):
        while len(x) > 0 and x[-1] >= cutoffs[i-1]:
            x.pop(-1)
            histogram[i] += 1
    histogram[0] += len(x)
    has_value = (histogram > threshold).nonzero()[0]
    return cutoffs[has_value[0]], cutoffs[has_value[-1]]

def mean(x, epsilon, delta, bounds=False):
    """
    This function computes the differentially private estimate of the average 
    using the noisy average approch from [1]. If the bounds are not provided 
    by the user, then they are estimated using the histogram approach from [2].

    [1] Differential Privacy: From Theory to Practice
    [2] https://github.com/google/differential-privacy/
    """
    x = np.array(x)
    if not bounds:
        epsilon = epsilon / 2.0
        bounds = approx_bounds(x, epsilon)
    low, high = bounds
    x = np.minimum(np.maximum(x, low), high)
    noise = np.random.laplace() * (high - low) / epsilon
    mean = (np.sum(x) + noise) / len(x)
    return min(max(low, mean), high)

def sum(x, epsilon, delta, bounds=False):
    """
    This function computes the differentially private estimate of the sum. If 
    the bounds are not provided by the user, then they are estimated using the 
    histogram approach from [1].

    [1] https://github.com/google/differential-privacy/
    """
    x = np.array(x)
    if not bounds:
        epsilon = epsilon / 2.0
        bounds = approx_bounds(x, epsilon)
    low, high = bounds
    x = np.minimum(np.maximum(x, low), high)
    noise = np.random.laplace() * (high - low) / epsilon
    return np.sum(x) + noise

def median(x, epsilon, delta):
    """
    This function computes the differentially private estimate of the median 
    using the approach proposed in [1]. It uses the Laplace mechanism on page 
    10 and the smooth sensitivity of the median derivation found on page 12.
    
    The resulting value is (epsilon-delta) differentially private. By default,
    if not specified, delta is set to `1/(100*len(x))` so that there is a 99%
    chance that differential privacy is satisfied for any individual.

    [1] http://www.cse.psu.edu/~ads22/pubs/NRS07/NRS07-full-draft-v1.pdf
    """
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

def median_gaussian(x, epsilon, delta):
    """
    This function computes the differentially private estimate of the median 
    using the approach proposed in [1]. It uses the Gaussian mechanism on page 
    10 and the smooth sensitivity of the median derivation found on page 12.

    The resulting value is (epsilon-delta) differentially private. By default,
    if not specified, delta is set to `1/(100*len(x))` so that there is a 99%
    chance that differential privacy is satisfied for any individual.

    [1] http://www.cse.psu.edu/~ads22/pubs/NRS07/NRS07-full-draft-v1.pdf
    """
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

def sample_and_aggregate(x, func, epsilon, nb_partitions, delta):
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
    return median(np.array(results), epsilon, delta)

def mean_sample_and_aggregate(x, epsilon, delta):
    """
    This function computes the differentially private estimate of the average 
    using the smooth sensitivity and sample and aggregate approach.
    """
    nb_partitions = int(np.sqrt(len(x)))
    return sample_and_aggregate(x, np.mean, epsilon, nb_partitions, delta)

def sum_sample_and_aggregate(x, epsilon, delta):
    """
    This function computes the differentially private estimate of the sum 
    using the smooth sensitivity and sample and aggregate approach.
    """
    nb_partitions = int(np.sqrt(len(x)))
    return nb_partitions * sample_and_aggregate(x, np.sum, epsilon, nb_partitions, delta)

def finite_categorical(x, domain, epsilon):
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

def bounded_continuous(x, low, high, epsilon):
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
