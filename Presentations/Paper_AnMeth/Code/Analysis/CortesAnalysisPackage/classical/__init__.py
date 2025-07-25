import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import simpson
from scipy.stats.mstats import linregress

def generic_area_under_peak(peak_func, x, *params):
    area = simpson(peak_func(x, *params), x)
    return area

def find_valley(lines, bins):
    _n = len(lines)
    _midpoint = int(_n / 2)
    _left = lines[:_midpoint]
    _right = lines[_midpoint:]
    _left = np.array(_left)
    _right = np.array(_right)
    left = np.min(_left)
    right = np.min(_right)
    
    left_index = np.where(_left == left)[0][0]
    left_bin = bins[left_index]
    
    right_index = np.where(_right == right)[0][0]
    right_index = right_index + _midpoint
    right_bin = bins[right_index]

    return left_bin, right_bin, left, right

def PerpendicularDrop(
    df: pd.DataFrame, 
    c_window, 
    si_window, 
    bins: None) -> tuple:

    if bins is None:
        bins = np.arange(0, len(df), 1)
    c_bins = bins[(bins >= c_window[0]) & (bins <= c_window[1])]
    si_bins = bins[(bins >= si_window[0]) & (bins <= si_window[1])]
    
    c_df = df.loc[(bins >= c_window[0]) & (bins <= c_window[1])]
    si_df = df.loc[(bins >= si_window[0]) & (bins <= si_window[1])]
    
    
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

        min_y = y_fit.min()
        c_peak_line = y_fit - min_y

        c_peak_area = simpson(c_peak_line, x_fit)
        c_peak_areas.append(c_peak_area)

        c_peak_fitting_err = (y_fit - c_peak_line).std()
        c_peak_fitting_errs.append(c_peak_fitting_err)
        
        c_baseline_lines.append([min_y]*len(x_fit))
        c_peak_lines.append(np.array(c_peak_line)+np.array([min_y]*len(x_fit)))

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
        min_y = y_fit.min()
        si_peak_line = y_fit - min_y
        si_peak_area = simpson(si_peak_line, x_fit)
        si_peak_areas.append(si_peak_area)
        si_peak_fitting_err = (y_fit - si_peak_line).std()
        si_peak_fitting_errs.append(si_peak_fitting_err)
        si_baseline_lines.append([min_y]*len(x_fit))
        si_peak_lines.append(np.array(si_peak_line)+np.array([min_y]*len(x_fit)))


    fitting_df = pd.DataFrame()
    fitting_df["label"] = df.columns
    fitting_df["Carbon Peak Area"] = c_peak_areas
    fitting_df["Carbon Peak Area Error"] = c_peak_fitting_errs
    fitting_df["Silicone Peak Area"] = si_peak_areas
    fitting_df["Silicone Peak Area Error"] = si_peak_fitting_errs

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

    return fitting_df, c_lines_df, si_lines_df

def TangentSkim(
    df: pd.DataFrame, 
    c_window, 
    si_window, 
    bins: None) -> tuple:

    if bins is None:
        bins = np.arange(0, len(df), 1)
    c_bins = bins[(bins >= c_window[0]) & (bins <= c_window[1])]
    si_bins = bins[(bins >= si_window[0]) & (bins <= si_window[1])]
    
    c_df = df.loc[(bins >= c_window[0]) & (bins <= c_window[1])]
    si_df = df.loc[(bins >= si_window[0]) & (bins <= si_window[1])]
    
    
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

        left_bin, right_bin, left_min, right_min = find_valley(y_fit, x_fit)
        # line between left and right valley
        slope, intercept, r_value, p_value, std_err = linregress(
            [left_bin, right_bin], 
            [left_min, right_min]
        )
        
        c_baseline_line = slope * x_fit + intercept
        c_baseline_lines.append(c_baseline_line)
        c_peak_line = y_fit - c_baseline_line
        # values to the left of the left valley are 0
        c_peak_line[:np.where(x_fit == left_bin)[0][0]] = 0
        # values to the right of the right valley are 0
        c_peak_line[np.where(x_fit == right_bin)[0][0]:] = 0
        c_peak_area = simpson(c_peak_line, x_fit)
        c_peak_areas.append(c_peak_area)
        c_peak_fitting_err = (y_fit - c_baseline_line).std()
        c_peak_fitting_errs.append(c_peak_fitting_err)
        c_peak_lines.append(np.array(c_peak_line)+np.array(c_baseline_line))

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
        left_bin, right_bin, left_min, right_min = find_valley(y_fit, x_fit)
        # line between left and right valley
        slope, intercept, r_value, p_value, std_err = linregress(
            [left_bin, right_bin], 
            [left_min, right_min]
        )
        si_baseline_line = slope * x_fit + intercept
        si_baseline_lines.append(si_baseline_line)
        si_peak_line = y_fit - si_baseline_line
        # values to the left of the left valley are 0
        si_peak_line[:np.where(x_fit == left_bin)[0][0]] = 0
        # values to the right of the right valley are 0
        si_peak_line[np.where(x_fit == right_bin)[0][0]:] = 0
        si_peak_area = simpson(si_peak_line, x_fit)
        si_peak_areas.append(si_peak_area)
        si_peak_fitting_err = (y_fit - si_baseline_line).std()
        si_peak_fitting_errs.append(si_peak_fitting_err)
        si_peak_lines.append(np.array(si_peak_line)+np.array(si_baseline_line))

    fitting_df = pd.DataFrame()
    fitting_df["label"] = df.columns
    fitting_df["Carbon Peak Area"] = c_peak_areas
    fitting_df["Carbon Peak Area Error"] = c_peak_fitting_errs
    fitting_df["Silicone Peak Area"] = si_peak_areas
    fitting_df["Silicone Peak Area Error"] = si_peak_fitting_errs

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

    return fitting_df, c_lines_df, si_lines_df
