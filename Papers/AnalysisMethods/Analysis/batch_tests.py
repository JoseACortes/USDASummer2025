# %% Imports

import pandas as pd
import numpy as np
from tabulate import tabulate

from CortesAnalysisPackage import plot as pt
from CortesAnalysisPackage import peakfitting as pfa
from CortesAnalysisPackage import componentfitting as ca
from CortesAnalysisPackage import classical as cla
from CortesAnalysisPackage import svd
from CortesAnalysisPackage import deeplearning as dl
from CortesAnalysisPackage import activation as act

from scipy.stats.mstats import linregress

# %%
# Load the data
datasets_folder = '../Datasets/'
DetectorReadings = pd.read_pickle(datasets_folder + 'DetectorReadings.pkl')
bins = DetectorReadings['bins'].values
DetectorReadings = DetectorReadings.drop(columns=['bins'])

ExpirementData = pd.read_pickle(datasets_folder + 'ExpirementData.pkl')

true_c_concentrations = ExpirementData['avg_carbon_portion'].values

# %% important data groups
Elems = ['C', 'Si', 'Al', 'H', 'Na', 'O', 'Fe', 'Mg']
Elems_mask = ExpirementData.function.isin(Elems)
ExpirementData['IsElement'] = ExpirementData.function.isin(Elems)

Compounds = ['SiO2', 'Al2O3', 'H2O', 'Na2O', 'Fe2O3', 'MgO']
Compounds_mask = ExpirementData.function.isin(Compounds)
ExpirementData['IsCompound'] = ExpirementData.function.isin(Compounds)

CommonSoils = ['Silica', 'Kaolinite', 'Smectite', 'Montmorillonite', 'Quartz', 'Chlorite', 'Mica', 'Feldspar']
CommonSoils_mask = ExpirementData.function.isin(CommonSoils)
ExpirementData['IsCommonSoil'] = ExpirementData.function.isin(CommonSoils)
CommonSoils_filenames = ExpirementData[CommonSoils_mask].filename.tolist()
CommonSoils_filenames = set(CommonSoils_filenames)

no_carbon_mask = ExpirementData['avg_carbon_portion']==0
no_carbon_filenames = ExpirementData[no_carbon_mask].filename.tolist()
no_carbon_filenames = set(no_carbon_filenames)

little_carbon_mask = (ExpirementData['avg_carbon_portion']<=0.06) & (ExpirementData['avg_carbon_portion']!=0)
little_carbon_filenames = ExpirementData[little_carbon_mask].filename.tolist()
little_carbon_filenames = set(little_carbon_filenames)

large_carbon_mask = ExpirementData['avg_carbon_portion']>0.6
large_carbon_filenames = ExpirementData[large_carbon_mask].filename.tolist()
large_carbon_filenames = set(large_carbon_filenames)

# %%
# Define train and test indices

# Set 1: All Data
training_set_one = ExpirementData.filename.tolist()

# Set 2: Common Soils and little Carbon
training_set_two = CommonSoils_filenames\
    .union(little_carbon_filenames)

# Set 3: Common Soils and large Carbon
training_set_three = CommonSoils_filenames\
    .union(large_carbon_filenames)

# Set 4: Common Soils and all Carbon
training_set_four = CommonSoils_filenames\
    .union(no_carbon_filenames)\
        .union(little_carbon_filenames)\
            .union(large_carbon_filenames)

training_set_names = [
    'All Data',
    'Common Soils and little Carbon',
    'Common Soils and large Carbon',
    'Common Soils and all Carbon'
]

# %% Results Setup
results = {
    'Test Name': [],
    'Prediction MSE': [],
    'Prediction R2': [],
    'Predicted Concentrations': []
}

# %% differnt test methods
def peakfitting_linear_baseline(df, bins, training_cols, true_c_concentrations):
    fitting_df, carbon_fitting_df, si_fitting_df, c_lines_df, si_lines_df = pfa.PeakFit(
        df,
        bins=bins,
        c_window=[4.2, 4.7], 
        si_window=[1.6, 1.95],
        c_baseline='linear',
        si_baseline='linear',
        )
    
    training_mask = fitting_df.label.isin(training_cols)
    training = fitting_df[training_mask]
    training_x = training['Carbon Peak Area']
    training_y = np.array(true_c_concentrations)[training_mask]

    testing = fitting_df
    testing_x = testing['Carbon Peak Area']
    testing_y = np.array(true_c_concentrations)

    method_name = 'PF - Linear Baseline'

    activation_name, act_results, mse_test, r2, pred_y = act.linearActivation(
        training_x,
        training_y,
        testing_x,
        testing_y,
        fitting_df['Carbon Peak Area']
    )

    return method_name, mse_test, r2, pred_y

def peakfitting_falloff_baseline(df, bins, training_cols, true_c_concentrations):
    fitting_df, carbon_fitting_df, si_fitting_df, c_lines_df, si_lines_df = pfa.PeakFit(
        df,
        bins=bins,
        c_window=[4.2, 4.7], 
        si_window=[1.6, 1.95],
        c_baseline='exp_falloff',
        si_baseline='exp_falloff',
        )
    
    training_mask = fitting_df.label.isin(training_cols)
    training = fitting_df[training_mask]
    training_x = training['Carbon Peak Area']
    training_y = np.array(true_c_concentrations)[training_mask]

    testing = fitting_df
    testing_x = testing['Carbon Peak Area']
    testing_y = np.array(true_c_concentrations)

    method_name = 'PF - Exp Falloff Baseline'

    activation_name, act_results, mse_test, r2, pred_y = act.linearActivation(
        training_x,
        training_y,
        testing_x,
        testing_y,
        fitting_df['Carbon Peak Area']
    )

    return method_name, mse_test, r2, pred_y

def component_analysis(df, bins, training_cols, true_c_concentrations):
    fitting_df, predicted_df = ca.Analyze(
        df,
        bins=bins,
        train_cols=training_cols,
        true_c_concentrations=true_c_concentrations,
        normalize=False,
        initial_guess=np.array([1/len(training_cols)]*len(training_cols)),
        window = [4.2, 4.7]
    )
    # get the training column with the highest carbon concentration
    
    training_cols_index = [df.columns.get_loc(col) for col in training_cols if col in fitting_df.columns]
    max_c_index = np.argmax(true_c_concentrations[training_cols_index])
    max_c_col = df.columns[training_cols_index[max_c_index]]

    training_mask = df.columns.isin(training_cols)
    training = fitting_df[training_mask]
    training_x = training[max_c_col]
    training_y = np.array(true_c_concentrations)[training_mask]

    testing = fitting_df
    testing_x = testing[max_c_col]
    testing_y = np.array(true_c_concentrations)

    method_name = 'CA - Component Analysis'

    activation_name, act_results, mse_test, r2, pred_y = act.linearActivation(
        training_x,
        training_y,
        testing_x,
        testing_y,
        fitting_df[max_c_col]
    )

    return method_name, mse_test, r2, pred_y

component_analysis(DetectorReadings, bins, list(training_set_four), true_c_concentrations)


# %%
# TODO: Add SVD, DL, and Classical methods. Move set characteristics to data augmentation. Include Convoluted Data.