import scipy.optimize as opt
import pandas as pd
import numpy as np

def linear_combination(x, *params):
    return sum(p * xi for p, xi in zip(params, x))

def residuals(params, x, y):
    return np.average((y - linear_combination(x, *params))**2)

def convex_residuals(params, x, y, alpha=1):
    residual = np.average((y - linear_combination(x, *params))**2)
    return residual + alpha*(1-np.sum(params))**2

def find_linear_combination(x, y, convex=False, initial_guess=None):
    if initial_guess is None:
        initial_guess = [1/x.shape[0]] * x.shape[0]
    to_minimize = residuals
    if convex:
        to_minimize = convex_residuals
    res = opt.minimize(to_minimize, x0=initial_guess, args=(x, y), bounds=([[0,1]]*x.shape[0]), tol=1e-16)
    return res.x

def Decompose(
    df,
    train_cols=None,
    normalize=False,
    convex_regression=False,
):
    # Calculate the linear combination of the columns
    x = np.array(df.values.T)
    if train_cols is None:
        train_cols = df.columns
    x_train = np.array(df[train_cols].values.T)

    coeffss = []
    for _x in x:
        coeffs = find_linear_combination(x_train, _x, convex=convex_regression)
        coeffss.append(coeffs)
    coeffs = np.array(coeffss)
    if normalize:
        
        coeffs = coeffs/np.sum(coeffs, axis=1)[:, np.newaxis]

    # Create a new DataFrame with the results
    fitting_df = pd.DataFrame(coeffs, index=df.columns, columns=train_cols)

    return fitting_df

def Analyze(
    df,
    bins=None,
    train_cols=None,
    true_c_concentrations=None,
    normalize=True,
    initial_guess=None,
    window = None,
):
    """
    Analyze the data frame and return a new data frame with the results.

    Parameters
    ----------
    df : pandas.DataFrame
        The input data frame.
    
    Returns
    -------
    pandas.DataFrame
        The new data frame with the results.
    """
    # Calculate the linear combination of the columns
    
    if bins is None:
        bins = np.arange(len(df))

    if window is None:
        window = [bins[0], bins[-1]]
    
    
    x = np.array(df.values.T)


    bin_index = np.array([i for i in range(len(bins)) if bins[i] >= window[0] and bins[i] <= window[1]])
    x = x[:, bin_index]

    if train_cols is None:
        x_train = x
        true_c_train = np.array(true_c_concentrations)
    elif isinstance(train_cols, list) and isinstance(train_cols[0], str):
        train_cols = [df.columns.get_loc(col) for col in train_cols if col in df.columns]
        x_train = x[train_cols, :]
        true_c_train = np.array(true_c_concentrations)[train_cols]
    elif isinstance(train_cols, list) and isinstance(train_cols[0], int):
        train_cols = [col for col in train_cols if col < len(df.columns)]
        x_train = x[train_cols, :]
        true_c_train = np.array(true_c_concentrations)[train_cols]
    # print(x_train.shape)

    coeffss = []
    for _x in x:
        coeffs = find_linear_combination(
            x_train, 
            _x,
            initial_guess=initial_guess)
        coeffss.append(coeffs)
    coeffs = np.array(coeffss)
    if normalize:
        coeffs = coeffs/np.sum(coeffs, axis=1)[:, np.newaxis]
    determined_c = np.sum(coeffs*true_c_train, axis=1)

    # Create a new DataFrame with the results
    fitting_df = pd.DataFrame(coeffs, index=df.columns, columns=df.columns[train_cols])
    predicted_df = pd.DataFrame(determined_c, index=df.columns, columns=['Carbon Portion'])

    return fitting_df, predicted_df

