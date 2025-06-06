import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import simpson
from scipy.stats.mstats import linregress

def linearfunc(x, a, b):
    return a * x + b

def gaussianfunc(x, a, b, c):
    return a * np.exp(-((x - b) / c) ** 2)

def double_gaussian(x, a1, b1, c1, a2, b2, c2):
    return gaussianfunc(x, a1, b1, c1) + gaussianfunc(x, a2, b2, c2)

def geb(x, a, b, c):
    return (a+b*np.sqrt(x+c*(x*x)))*0.60056120439322

def exp_falloff(x,x0,a,p,b):
    return (a*np.exp(-p*(x-x0)))+b

def lorentzian(x, x0, a, b):
    return 1 / (a*(x - x0) ** 2 + b ** 2 + 1)

functions = {
    'linear': linearfunc,
    'gauss': gaussianfunc,
    'double_gauss': double_gaussian,
    'exp_falloff': exp_falloff,
    'lorentzian': lorentzian,
}

gaus_fix_term = 0.60056120439322

def generate_initial_guess(df, bins, f, p0='auto'):
    if p0 == 'auto':
        if f == linearfunc:
            p0 = [0, df.min().min()]
        elif f == gaussianfunc:
            p0 = [gaus_fix_term*(df.max().max()-df.min().min()), np.mean(bins), (bins[-1]-bins[0])/6]
        elif f == double_gaussian:
            p0 = [gaus_fix_term*(df.max().max()-df.min().min())/2, np.mean(bins), (bins[-1]-bins[0])/6, gaus_fix_term*(df.max().max()-df.min().min())/2, np.mean(bins), (bins[-1]-bins[0])/ 6]
        elif f == exp_falloff:
            p0 = [np.min(bins), gaus_fix_term*(df.max().max()-df.min().min()), 1, df.min().min()]
        elif f == lorentzian:
            p0 = [0, 0, 0]
    return p0

def generate_bounds(df, bins, f, bounds='auto'):
    if bounds == 'auto':
        if f == linearfunc:
            bounds = ([-np.inf, 0], [0, np.inf])
        elif f == gaussianfunc:
            bounds = ([0, np.min(bins), (bins[-1]-bins[0])/100], [(df.max().max()-df.min().min()), np.max(bins), (bins[-1]-bins[0])/2])
        elif f == double_gaussian:
            bounds = ([0, np.min(bins), (bins[-1]-bins[0])/100, 0, np.min(bins), (bins[-1]-bins[0])/100], [(df.max().max()-df.min().min()), np.max(bins), (bins[-1]-bins[0])/2, (df.max().max()-df.min().min()), np.max(bins), (bins[-1]-bins[0])/2])
        elif f == exp_falloff:
            bounds = ([-np.inf]*4, [np.inf]*4)
        elif f == lorentzian:
            bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
    return bounds

def generic_area_under_peak(peak_func, x, *params):
    area = simpson(peak_func(x, *params), x)
    return area

def PeakFit(
    df: pd.DataFrame, 
    c_window, 
    si_window, 
    bins: None, 
    c_baseline= 'linear', 
    c_peak = 'gauss', 
    si_baseline = 'linear', 
    si_peak = 'gauss', 
    c_peak_p0 = 'auto', 
    si_peak_p0 = 'auto', 
    c_peak_bounds = 'auto', 
    si_peak_bounds = 'auto', 
    c_baseline_p0 = 'auto', 
    si_baseline_p0 = 'auto', 
    c_baseline_bounds = 'auto', 
    si_baseline_bounds = 'auto') -> tuple:
    """
    Analyze spectral data to fit carbon and silicon peaks and baselines.
    Parameters:
    df (pd.DataFrame): DataFrame containing spectral data.
    c_window (tuple): Tuple specifying the carbon window range (start, end).
    si_window (tuple): Tuple specifying the silicon window range (start, end).
    bins (array-like, optional): X-axis values for the spectra. Defaults to None.
    c_baseline (str or function, optional): Baseline function or string identifier for carbon. Defaults to 'linear'.
    c_peak (str or function, optional): Peak function or string identifier for carbon. Defaults to 'gauss'.
    si_baseline (str or function, optional): Baseline function or string identifier for silicon. Defaults to 'linear'.
    si_peak (str or function, optional): Peak function or string identifier for silicon. Defaults to 'gauss'.
    c_peak_p0 (array-like or str, optional): Initial guess for carbon peak parameters or 'auto'. Defaults to 'auto'.
    si_peak_p0 (array-like or str, optional): Initial guess for silicon peak parameters or 'auto'. Defaults to 'auto'.
    c_peak_bounds (tuple or str, optional): Bounds for carbon peak parameters or 'auto'. Defaults to 'auto'.
    si_peak_bounds (tuple or str, optional): Bounds for silicon peak parameters or 'auto'. Defaults to 'auto'.
    c_baseline_p0 (array-like or str, optional): Initial guess for carbon baseline parameters or 'auto'. Defaults to 'auto'.
    si_baseline_p0 (array-like or str, optional): Initial guess for silicon baseline parameters or 'auto'. Defaults to 'auto'.
    c_baseline_bounds (tuple or str, optional): Bounds for carbon baseline parameters or 'auto'. Defaults to 'auto'.
    si_baseline_bounds (tuple or str, optional): Bounds for silicon baseline parameters or 'auto'. Defaults to 'auto'.
    Returns:
    tuple: A tuple containing the following DataFrames:
    - fitting_df (pd.DataFrame): DataFrame with peak areas and fitting errors.
    - carbon_fitting_df (pd.DataFrame): DataFrame with carbon fitting parameters.
    - si_fitting_df (pd.DataFrame): DataFrame with silicon fitting parameters.
    - c_lines_df (pd.DataFrame): DataFrame with carbon baseline and peak lines.
    - si_lines_df (pd.DataFrame): DataFrame with silicon baseline and peak lines.
    """
    
    if bins is None:
        bins = np.arange(0, len(df), 1)
    c_bins = bins[(bins >= c_window[0]) & (bins <= c_window[1])]
    si_bins = bins[(bins >= si_window[0]) & (bins <= si_window[1])]
    
    c_df = df.loc[(bins >= c_window[0]) & (bins <= c_window[1])]
    si_df = df.loc[(bins >= si_window[0]) & (bins <= si_window[1])]

    if isinstance(c_baseline, str):
        c_baseline = functions[c_baseline]
    c_baseline_p0 = generate_initial_guess(c_df, c_bins, c_baseline, p0=c_baseline_p0)
    c_baseline_lower, c_baseline_upper = generate_bounds(c_df, c_bins, c_baseline, bounds=c_baseline_bounds)

    if isinstance(si_baseline, str):
        si_baseline = functions[si_baseline]
    si_baseline_p0 = generate_initial_guess(si_df, si_bins, si_baseline, p0=si_baseline_p0)
    si_baseline_lower, si_baseline_upper = generate_bounds(si_df, si_bins, si_baseline, bounds=si_baseline_bounds)

    if isinstance(c_peak, str):
        c_peak = functions[c_peak]
    c_peak_p0 = generate_initial_guess(c_df, c_bins, c_peak, p0=c_peak_p0)
    c_peak_lower, c_peak_upper = generate_bounds(c_df, c_bins, c_peak, bounds=c_peak_bounds)

    if isinstance(si_peak, str):
        si_peak = functions[si_peak]
    si_peak_p0 = generate_initial_guess(si_df, si_bins, si_peak, p0=si_peak_p0)
    si_peak_lower, si_peak_upper = generate_bounds(si_df, si_bins, si_peak, bounds=si_peak_bounds)
    
    c_p0 = c_baseline_p0 + c_peak_p0
    c_lower = c_baseline_lower + c_peak_lower
    c_upper = c_baseline_upper + c_peak_upper
    c_bounds = (c_lower, c_upper)

    si_p0 = si_baseline_p0 + si_peak_p0
    si_lower = si_baseline_lower + si_peak_lower
    si_upper = si_baseline_upper + si_peak_upper
    si_bounds = (si_lower, si_upper)

    def total_c(x, *params):
        return c_baseline(x, *params[:len(c_baseline_p0)]) + c_peak(x, *params[len(c_baseline_p0):])
    def total_si(x, *params):
        return si_baseline(x, *params[:len(si_baseline_p0)]) + si_peak(x, *params[len(si_baseline_p0):])
    
    
    c_popts = []
    c_pcovs = []
    c_infodicts = []
    c_mesgs = []
    c_iers = []
    c_peak_areas = []
    c_peak_fitting_errs = []
    c_baseline_lines = []
    c_peak_lines = []

    for col in c_df.columns:
        x_fit = c_bins
        y_fit = c_df[col]
        p0 = np.clip(c_p0, c_lower, c_upper)
        popt, pcov, infodict, mesg, ier = curve_fit(
            total_c, 
            x_fit, 
            y_fit, 
            p0=p0,
            bounds=c_bounds,
            full_output=True,
            maxfev=100000,
            ftol=1e-15
        )
        c_popts.append(popt)
        c_pcovs.append(pcov)
        c_infodicts.append(infodict)
        c_mesgs.append(mesg)
        c_iers.append(ier)

        # c_peak_area = area_under_gaussian_peak(*popt[len(c_baseline_p0):]) # will need to change this if the peak function changes
        c_peak_area = generic_area_under_peak(c_peak, x_fit, *popt[len(c_baseline_p0):])
        c_peak_areas.append(c_peak_area)

        c_peak_fitting_err = (y_fit - total_c(x_fit, *popt)).std()
        c_peak_fitting_errs.append(c_peak_fitting_err)
        
        c_baseline_lines.append(c_baseline(x_fit, *popt[:len(c_baseline_p0)]))
        c_peak_lines.append(total_c(x_fit, *popt))

    # silicon fitting    
    si_popts = []
    si_pcovs = []
    si_infodicts = []
    si_mesgs = []
    si_iers = []
    si_peak_areas = []
    si_peak_fitting_errs = []
    si_baseline_lines = []
    si_peak_lines = []

    for col in si_df.columns:
        x_fit = si_bins
        y_fit = si_df[col]
        # Ensure initial guess p0 is within bounds
        p0 = np.clip(si_p0, si_lower, si_upper)
        popt, pcov, infodict, mesg, ier = curve_fit(
            total_si, 
            x_fit, 
            y_fit, 
            p0=p0,
            bounds=si_bounds,
            full_output=True,
            maxfev=100000
        )
        si_popts.append(popt)
        si_pcovs.append(pcov)
        si_infodicts.append(infodict)
        si_mesgs.append(mesg)
        si_iers.append(ier)

        # si_peak_area = area_under_gaussian_peak(*popt[len(si_baseline_p0):]) # will need to change this if the peak function changes
        si_peak_area = generic_area_under_peak(si_peak, x_fit, *popt[len(si_baseline_p0):])
        si_peak_areas.append(si_peak_area)

        si_peak_fitting_err = (y_fit - total_si(x_fit, *popt)).std()
        si_peak_fitting_errs.append(si_peak_fitting_err)

        si_baseline_lines.append(si_baseline(x_fit, *popt[:len(si_baseline_p0)]))
        si_peak_lines.append(total_si(x_fit, *popt))

    fitting_df = pd.DataFrame()
    fitting_df["label"] = df.columns
    fitting_df["Carbon Peak Area"] = c_peak_areas
    fitting_df["Carbon Peak Area Error"] = c_peak_fitting_errs
    fitting_df["Silicone Peak Area"] = si_peak_areas
    fitting_df["Silicone Peak Area Error"] = si_peak_fitting_errs

    carbon_fitting_df = pd.DataFrame()
    carbon_fitting_df["labels"] = df.columns
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i in range(len(c_baseline_p0)):
        np.array(c_popts).shape
        carbon_fitting_df[f"Baseline {alphabet[i]}"] = np.array(c_popts)[:, i]
    for i in range(len(c_peak_p0)):
        carbon_fitting_df[f"Peak {alphabet[i]}"] = np.array(c_popts)[:, i+len(c_baseline_p0)]

    si_fitting_df = pd.DataFrame()
    si_fitting_df["labels"] = df.columns
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i in range(len(si_baseline_p0)):
        np.array(si_popts).shape
        si_fitting_df[f"Baseline {alphabet[i]}"] = np.array(si_popts)[:, i]
    for i in range(len(si_peak_p0)):
        si_fitting_df[f"Peak {alphabet[i]}"] = np.array(si_popts)[:, i+len(si_baseline_p0)]

    c_lines_df = pd.DataFrame()
    c_lines_df["bins"] = c_bins
    for i in range(len(df.columns)):
        c_lines_df[df.columns[i]+" baseline"] = c_baseline_lines[i]
        c_lines_df[df.columns[i]+" peak"] = c_peak_lines[i]
        c_lines_df[df.columns[i]+" true"] = c_df[df.columns[i]].values
    
    si_lines_df = pd.DataFrame()
    si_lines_df["bins"] = si_bins
    for i in range(len(df.columns)):
        si_lines_df[df.columns[i]+" baseline"] = si_baseline_lines[i]
        si_lines_df[df.columns[i]+" peak"] = si_peak_lines[i]
        si_lines_df[df.columns[i]+" true"] = si_df[df.columns[i]].values

    return fitting_df, carbon_fitting_df, si_fitting_df, c_lines_df, si_lines_df

# def Analyze(
#         training_x,
#         testing_x,
#         training_y,
#         testing_y
# ):
#     """
#     Analyze the training and testing data.
#     Parameters:
#     training_x (pd.DataFrame): Training data features.
#     testing_x (pd.DataFrame): Testing data features.
#     training_y (pd.Series): Training data labels.
#     testing_y (pd.Series): Testing data labels.
#     Returns:
#     tuple: A tuple containing the following DataFrames:
#     - fitting_df (pd.DataFrame): DataFrame with peak areas and fitting errors.
#     - carbon_fitting_df (pd.DataFrame): DataFrame with carbon fitting parameters.
#     - si_fitting_df (pd.DataFrame): DataFrame with silicon fitting parameters.
#     - c_lines_df (pd.DataFrame): DataFrame with carbon baseline and peak lines.
#     - si_lines_df (pd.DataFrame): DataFrame with silicon baseline and peak lines.
#     """
#     # linear regression
#     slope, intercept, r, p, se = linregress(training_x, training_y)
#     analyze = lambda x: slope * x + intercept
#     mae_test = np.mean(np.abs(testing_y - analyze(testing_x)))