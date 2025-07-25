import numpy as np
from scipy.optimize import nnls
import pandas as pd
import tqdm

def SVD(
    df: pd.DataFrame, 
    target_window,
    weak_window= None,
    bins= None,
    geb_a = -0.026198,
    geb_b = 0.059551,
    geb_c = -0.037176,
    ):

    if bins is None:
        bins = np.arange(0, len(df), 1)

    if weak_window is None:
        weak_window = [0, len(df) - 1]

    weak_bins = bins[(bins >= weak_window[0]) & (bins <= weak_window[1])]
    weak_df = df.loc[(bins >= weak_window[0]) & (bins <= weak_window[1])]
    
    
    Y = weak_df.values
    num_bins = Y.shape[0]
    num_channels = num_bins  # or set as needed
    R = np.eye(num_bins)


    # Calculate FWHM for each energy bin
    FWHM = geb_a + geb_b * np.sqrt(weak_bins) + geb_c * weak_bins**2
    sigma = FWHM / 2.355 
    sigma = np.abs(sigma) 

    R = np.zeros((num_bins, num_bins))
    for j in range(num_bins):  # Loop over columns (channels)
        # Gaussian response for energy bin i and channel j
        E_j = weak_bins[j]
        sigma_j = sigma[j]
        
        # Discretize integral using small energy steps
        E_steps = weak_bins
        gaussian = (1 / (np.sqrt(2 * np.pi) * sigma_j)) * np.exp(-((E_steps - E_j)**2) / (2 * sigma_j**2))
        gaussian /= np.sum(gaussian)  # Normalize the Gaussian

        R[j, :] = gaussian
    
    v = np.ones(num_bins)  # or v = 1.0 / (Y + 1e-6) to avoid division by zero

    # 4. Weighted least squares: minimize sum_i v_i * (sum_j R_ij x_j - y_i)^2, x_j >= 0
    # This can be solved by transforming to standard NNLS:
    # sqrt(V) * R * x = sqrt(V) * Y
    sqrt_v = np.sqrt(v)
    R_weighted = R * sqrt_v[:, np.newaxis]
    Y_weighted = Y * sqrt_v[:, np.newaxis] if Y.ndim > 1 else Y * sqrt_v

    # 5. Solve for each spectrum (if multiple)
    X_est = []
    for i in tqdm.tqdm(range(Y_weighted.shape[1])):
        x, _ = nnls(
            R_weighted, 
            Y_weighted[:, i],
            atol=1e-12,  # Set tolerance for convergence
            )
        X_est.append(x)
    X_est = np.array(X_est)

    target_energy_bins = np.argwhere((weak_bins > target_window[0]) & (weak_bins < target_window[1]))

    _X = X_est[:, target_energy_bins]
    X = _X.sum(axis=1)

    fitting_df = pd.DataFrame(X, index=df.columns, columns=['Carbon Peak Area'])
    
    decomposed_df = pd.DataFrame(X_est, index=df.columns, columns=np.array(weak_bins).flatten())

    responses = pd.DataFrame(R, index=weak_bins, columns=weak_bins)

    # return X, X_est, R
    return fitting_df, decomposed_df, responses
