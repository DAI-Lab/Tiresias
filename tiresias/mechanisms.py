import numpy as np

def laplace_mechanism(x, sensitivity, epsilon=0.1):
    return np.random.laplace(loc=x, scale=sensitivity/epsilon)

def dp_median(x, epsilon):
    """
    This function computes the differentially private estimate of the median using the
    smooth sensitivity approach proposed in [1].

    [1] http://www.cse.psu.edu/~ads22/pubs/NRS07/NRS07-full-draft-v1.pdf
    """
    x = x.copy()
    x.sort()
    m = (len(x) + 1) // 2
    smooth_sensitivity = max(np.exp(-k * epsilon) * max(x[m+t] - x[m+t-k-1] for t in range(min(k, len(x)-m))) for k in range(1, len(x)))
    return laplace_mechanism(np.median(x), sensitivity=smooth_sensitivity, epsilon=epsilon)

def dp_mean(x, epsilon):
    """
    This function computes the differentially private estimate of the mean using the
    smooth sensitivity and sample and aggregate approach proposed in [1].

    [1] http://www.cse.psu.edu/~ads22/pubs/NRS07/NRS07-full-draft-v1.pdf
    """
    x = x.copy()
    np.random.shuffle(x)
    x = np.array([np.mean(partition) for partition in np.array_split(x, 5)])
    return dp_median(x, epsilon)

def dp_sum(x, epsilon):
    """
    This function computes the differentially private estimate of the sum using the
    smooth sensitivity and sample and aggregate approach proposed in [1].

    [1] http://www.cse.psu.edu/~ads22/pubs/NRS07/NRS07-full-draft-v1.pdf
    """
    x = x.copy()
    np.random.shuffle(x)
    x = np.array([np.sum(partition) for partition in np.array_split(x, 5)])
    return dp_median(x, epsilon) * 5
