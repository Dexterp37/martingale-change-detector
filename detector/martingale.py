import numpy as np

from scipy import interpolate
from scipy.spatial.distance import cosine

def normalize(hist):
    hist = hist.astype('float32')
    norm = np.sqrt(np.sum(hist**2))
    if norm == 0:
        return hist
    else:
        return hist / norm


def approx_equal(a, b, tol=10**-6):
    # We implement this function instead of using np.isclose as the latter is slow.
    return np.abs(a - b) < tol


def strangeness_cluster(x, samples):
    cluster_center = np.mean(samples, axis=0)
    return np.sqrt(np.sum(np.square(x - cluster_center)))


def representative_sample(samples):
    # Compute a representative histogram.
    return normalize(np.sum(samples, axis=0))


def compute_bhattacharyya_coefficient(x, y):
    # Compute the Bhattacharyya between two histograms.
    # The Bhattacharyya coefficient will be 0 if there is no
    # overlap at all due to the multiplication by zero in every
    # partition. This means the distance between fully separated
    # samples will not be exposed by this coefficient alone.
    return np.sum(np.sqrt(x * y))


def strangeness_bhattacharyya(x, samples):
    # This will only work with normalized histograms.
    omega_hist = representative_sample(samples)
    bc = compute_bhattacharyya_coefficient(x, omega_hist)
    return -np.log(bc) if bc > 0.0 else 0


def strangeness_hellinger(x, samples):
    # This will only work with normalized histograms.
    omega_hist = representative_sample(samples)
    bc = compute_bhattacharyya_coefficient(x, omega_hist)
    return np.sqrt(1. - bc) if bc < 1.0 else 1.0


def strangeness_cosine(x, samples):
    omega_hist = representative_sample(samples)
    return cosine(x, omega_hist)


# A dictionary of the supported strangeness functions, for easier consumptions
# from other modules.
SUPPORTED_STRANGENESS = {
    'cluster': strangeness_cluster,
    'bhattacharyya': strangeness_bhattacharyya,
    'hellinger': strangeness_hellinger,
    'cosine': strangeness_cosine
}

def martingale_test(data, threshold, strangeness_fun, epsilon=0.98, bootstrap_size=50):
    """ This function implements the martingale test with customizable
    strangeness functions from the "Martingale Framework for Detecting Changes in Data Streams
    by testing exchangeability" by Shyang Ho et al.
    """

    m_prev = 1.0
    changes = []
    martingales = []
    p_values = []
    strangeness = []
    period_index = 0

    # The algorithm needs a bootstrapping phase to work properly.
    bootstrap_stop_index = bootstrap_size

    for idx, x in enumerate(data):
        # Compute the current strangeness value
        x_strangeness = strangeness_fun(x, data[period_index:idx + 1]) if period_index != idx else 0.0
        strangeness.append(x_strangeness)

        # Theta should be randomly picked in [0, 1] but random()
        # uses an half-open interval, [0, 1). However, the probability
        # of picking exactly 1 should be 0, and it also pretty much depends
        # on the floating point implementation precision. So we should be
        # safe to use |random()|.
        theta_i = np.random.random()

        # Compute the p-value of the current strangeness value
        num_greater_strangeness = 0
        num_equal_strangeness = 0

        for s in strangeness[period_index:]:
            if approx_equal(s, x_strangeness):
                num_equal_strangeness += 1
            elif s > x_strangeness:
                num_greater_strangeness += 1

        p_i = (num_greater_strangeness + theta_i * num_equal_strangeness) / (idx - period_index + 1.0)
        p_i = p_i if p_i > 0 else 1 / (idx - period_index + 1.0)
        p_values.append(p_i)

        # Compute the value of the randomized power martingale.
        if idx < bootstrap_stop_index:
            m_curr = 1.0
        else:
            m_curr = (epsilon * (p_i ** (epsilon - 1.0)) * m_prev)
        martingales.append(m_curr)

        if m_curr > threshold:
            # Signal a change
            period_index = idx + 1
            bootstrap_stop_index = idx + bootstrap_size
            changes.append(idx)
            m_curr = 1.0

        # Store the value of the current martingale for the next iteration.
        m_prev = m_curr

    return changes, martingales, p_values, strangeness


def detect_changes(histograms, histogram_name,
                   strangeness_func=strangeness_cosine,
                   augmentation_factor=9,
                   threshold=20,
                   bootstrap_size=50,
                   normalize=True):
    hist_data = histograms[histogram_name]

    # Get the data for this histogram.
    data = np.array([h['values'] for h in hist_data], dtype="float32")

    # Augment the data
    x = np.arange(data.shape[0])
    y = hist_data[0]['buckets']
    xx = np.linspace(x.min(),x.max(), augmentation_factor * len(x))
    kernel = interpolate.RectBivariateSpline(x, y, data, kx=5, ky=1)

    data = kernel(xx, y)

    # Filter negative values:
    data[data < 0] = 0

    # Normalize the data to remove seasonalities
    if normalize:
        norms = np.sqrt((data**2).sum(axis=1, keepdims=True))
        norms[norms == 0] = 1
        data = data/norms

    # Detect changes in the histogram
    changes, martingale, pvalues, strangeness =\
        martingale_test(data, threshold, strangeness_func, bootstrap_size=bootstrap_size)

    return changes, martingale, pvalues, strangeness, data
