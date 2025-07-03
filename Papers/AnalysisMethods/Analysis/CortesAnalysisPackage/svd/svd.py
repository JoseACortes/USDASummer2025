import numpy as np
from scipy.optimize import nnls

def svd_unfolding(df, bins, geb_a=-0.026198, geb_b=0.059551, geb_c=-0.037176):
    """
    Perform SVD-based spectral unfolding using a Gaussian response matrix and NNLS.
    Args:
        df: DataFrame of measured spectra (shape: [num_bins, num_spectra])
        bins: 1D array of energy bins
        geb_a, geb_b, geb_c: GEB coefficients for FWHM calculation
    Returns:
        X_est: unfolded spectra (shape: [num_spectra, num_bins])
        R: response matrix
    """
    Y = df.values
    num_bins = Y.shape[0]
    # Calculate FWHM and sigma for each bin
    FWHM = geb_a + geb_b * np.sqrt(bins) + geb_c * bins**2
    sigma = np.abs(FWHM / 2.355)
    # Build response matrix R
    R = np.zeros((num_bins, num_bins))
    for j in range(num_bins):
        E_j = bins[j]
        sigma_j = sigma[j]
        gaussian = (1 / (np.sqrt(2 * np.pi) * sigma_j)) * np.exp(-((bins - E_j) ** 2) / (2 * sigma_j ** 2))
        gaussian /= np.sum(gaussian)
        R[j, :] = gaussian
    # Weighted least squares
    v = np.ones(num_bins)
    sqrt_v = np.sqrt(v)
    R_weighted = R * sqrt_v[:, np.newaxis]
    Y_weighted = Y * sqrt_v[:, np.newaxis] if Y.ndim > 1 else Y * sqrt_v
    # NNLS for each spectrum
    X_est = []
    for i in range(Y_weighted.shape[1]):
        x, _ = nnls(R_weighted, Y_weighted[:, i], atol=1e-12)
        X_est.append(x)
    X_est = np.array(X_est)
    return X_est, R