# %% [markdown]
# # Imports

# %%
import pandas as pd
import numpy as np
from tabulate import tabulate

from CortesAnalysisPackage import plot as pt
from CortesAnalysisPackage import peakfitting as pfa
from CortesAnalysisPackage import componentfitting as ca
from CortesAnalysisPackage import classical as cla
from CortesAnalysisPackage import svd as svd
# Import
import pandas as pd
import numpy as np
import INS_Analysis as insd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.mstats import linregress

# %%
import warnings

# %%
from CortesAnalysisPackage import ml as ml

# %% [markdown]
# # Data

# %%
df = pd.read_pickle("../Data/DetectorReadings.pkl")
bins = df['bins'].values
df = df.drop(columns=['bins'])
df.index = bins
exp_df = pd.read_pickle('../Data/ExpirementData.pkl')

# %%
# exp_df = exp_df[['filename', 'avg_carbon_portion', 'function','elem_maps']]
exp_df = exp_df.set_index('filename')

# %%
material_names = ['Silica', 'Kaolinite', 'Smectite', 'Montmorillonite', 'Quartz', 'Chlorite', 'Mica', 'Feldspar', 'Coconut']

# %%
material_mixes_exp_df = exp_df[exp_df.index.str.contains('|'.join(material_names), case=False)]
material_mixes_df = df[material_mixes_exp_df.index]
material_mixes_df.index = bins

# %%
elem_names = ['Si', 'Al', 'H', 'Na', 'O', 'Fe', 'Mg', 'C']
elem_exp_df = exp_df[exp_df['function'].isin(elem_names)]
elem_df = df[elem_exp_df.index]
elem_df.index = bins

# %%
elem_exp_df

# %%
exp_df

# %%
convolution_df = pd.read_pickle('../ConvolutedData/ConvolutionData.pkl')
convolution_exp_df = pd.read_pickle('../ConvolutedData/ConvolutionExpData.pkl')

# %%
convolution_exp_df = convolution_exp_df.set_index('filename')

# %%
convolution_exp_df

# %% [markdown]
# # Methods

# %% [markdown]
# ## Supplementary Code for Peak Fitting and Component Analysis

# %%
def lev_filter(df, exp_df, min, max):
    """
    Filters the DataFrame based on the average carbon portion in exp_df.
    """
    filtered_exp_df = exp_df[(exp_df['avg_carbon_portion'] >= min) & (exp_df['avg_carbon_portion'] <= max)]
    filtered_df = df[filtered_exp_df.index]
    return filtered_df, filtered_exp_df



def edge_split_train(df, exp_df):
    """
    Splits the DataFrame into training and testing sets with edge cases min and max carbon levels.
    """
    min_carbon = exp_df['avg_carbon_portion'].min()
    max_carbon = exp_df['avg_carbon_portion'].max()

    train_exp_df = exp_df[(exp_df['avg_carbon_portion'] == min_carbon) | (exp_df['avg_carbon_portion'] == max_carbon)]
    train_df = df[train_exp_df.index]

    return train_df, train_exp_df

def edge_split_test(df, exp_df):
    """
    Splits the DataFrame into inner training and testing sets with edge cases min and max carbon levels.
    """
    min_carbon = exp_df['avg_carbon_portion'].min()
    max_carbon = exp_df['avg_carbon_portion'].max()

    test_exp_df = exp_df[(exp_df['avg_carbon_portion'] > min_carbon) & (exp_df['avg_carbon_portion'] < max_carbon)]
    test_df = df[test_exp_df.index]

    return test_df, test_exp_df


# %%
def edge_split_train_avg(df, exp_df):
    """
    Splits the DataFrame into training and testing sets with edge cases min and max carbon levels.
    """

    train_exp_df = pd.DataFrame(columns=['filename', 'avg_carbon_portion'])
    train_df = pd.DataFrame()
    
    min_carbon = exp_df['avg_carbon_portion'].min()
    max_carbon = exp_df['avg_carbon_portion'].max()

    min_train_exp_df = exp_df[exp_df['avg_carbon_portion'] == min_carbon]
    min_train_df = df[min_train_exp_df.index]

    avg_min_carbon = min_train_exp_df['avg_carbon_portion'].mean()
    avg_min_carbon_filename = "average_min_carbon"
    
    train_exp_df = pd.concat([train_exp_df, pd.DataFrame([{'filename': avg_min_carbon_filename, 'avg_carbon_portion': avg_min_carbon}])], ignore_index=True)
    
    avg_min_carbon_spec = min_train_df.mean(axis=1)
    train_df = pd.concat([train_df, pd.DataFrame(avg_min_carbon_spec, columns=["average_min_carbon"])], axis=1)
    max_train_exp_df = exp_df[exp_df['avg_carbon_portion'] == max_carbon]
    max_train_df = df[max_train_exp_df.index]

    avg_max_carbon = max_train_exp_df['avg_carbon_portion'].mean()
    avg_max_carbon_filename = "average_max_carbon"
    train_exp_df = pd.concat([train_exp_df, pd.DataFrame([{'filename': avg_max_carbon_filename, 'avg_carbon_portion': avg_max_carbon}])], ignore_index=True)
    train_exp_df = train_exp_df.set_index('filename')
    avg_max_carbon_spec = max_train_df.mean(axis=1)
    
    train_df = pd.concat([train_df, pd.DataFrame(avg_max_carbon_spec, columns=["average_max_carbon"])], axis=1)
    
    return train_df, train_exp_df


# %%
just_feldspar_df = material_mixes_df.filter(like='Feldspar', axis=1)
just_feldspar_exp_df = material_mixes_exp_df.loc[just_feldspar_df.columns]

# %%
just_feldspar_exp_df

# %%
datasets = {
    "Material Mixes":
        {
            "train_df": material_mixes_df,
            "train_exp_df": material_mixes_exp_df,
            "test_df": material_mixes_df,
            "test_exp_df": material_mixes_exp_df,
            "name": "Material Mixes",
            "train_filter": "edge"
        },
    "Convolution Training":
        {
            "train_df": convolution_df,
            "train_exp_df": convolution_exp_df,
            "test_df": material_mixes_df,
            "test_exp_df": material_mixes_exp_df,
            "name": "Convolution",
            "train_filter": "all"
        },
    "Feldspar":
        {
            "train_df": just_feldspar_df,
            "train_exp_df": just_feldspar_exp_df,
            "test_df": just_feldspar_df,
            "test_exp_df": just_feldspar_exp_df,
            "name": "Feldspar",
            "train_filter": "edge"
        }
}

# %%
convolution_df

# %%
convolution_exp_df

# %%
Carbon_Levels = {
    'All': {
        'min': 0.0,
        'max': 1.0
    },
    'Agricultural': {
        'min': 0.00,
        'max': 0.06
    }
}

# %%
def activation_layer(target_val, true_carbon):
    # if all target_val are the same, return the average of true_carbon
    print("Target Values:", target_val)
    print("traget val type:", type(target_val))
    if np.all(target_val == target_val[0]):
        print('all target values are the same')
        mean_val = np.mean(true_carbon)
        result = linregress([0, 1], [mean_val, mean_val])
        reg = lambda x: result.slope * x + result.intercept
        analyze = lambda x: np.clip(reg(x), 0, 1)
    else:
        print('target values are not the same')
        result = linregress(target_val, true_carbon)
        reg = lambda x: result.slope * x + result.intercept
        analyze = lambda x: np.clip(reg(x), 0, 1)
    return result, analyze

# %% [markdown]
# ## 1. Peak Fitting

# %%
def PF_Linear(training_df, test_df, training_exp_df, test_exp_df):
    """
    Perform peak fitting on the training and test data.
    """
    training_cols = training_df.columns
    test_cols = test_df.columns
    
    combined_exp_df = pd.concat([training_exp_df, test_exp_df], axis=0)
    combined_exp_df = combined_exp_df.loc[~combined_exp_df.index.duplicated(keep='first')]

    combined_df = pd.concat([training_df, test_df], axis=1)
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated(keep='first')]
    
    true_carbon = combined_exp_df['avg_carbon_portion'].values
    test_true_carbon = test_exp_df['avg_carbon_portion'].values
    training_true_carbon = training_exp_df['avg_carbon_portion'].values
    
    fitting_df, carbon_fitting_df, si_fitting_df, c_lines_df, si_lines_df = pfa.PeakFit(
        combined_df,
        bins=combined_df.index,
        c_window=[4.2, 4.7], 
        si_window=[4.2, 4.7],
        c_baseline='linear',
        si_baseline='linear',
        )
    
    fitting_df = fitting_df.set_index('label')
    training_fitting_df = fitting_df.loc[training_cols]
    test_fitting_df = fitting_df.loc[test_cols]
    
    peak_areas = fitting_df['Carbon Peak Area'].values
    training_peak_areas = training_fitting_df['Carbon Peak Area'].values
    test_peak_areas = test_fitting_df['Carbon Peak Area'].values
    
    result, analyze = activation_layer(training_peak_areas, training_true_carbon)
    
    predicted_carbon = fitting_df['Carbon Peak Area'].apply(analyze).values
    predicted_training_carbon = training_fitting_df['Carbon Peak Area'].apply(analyze).values
    predicted_test_carbon = test_fitting_df['Carbon Peak Area'].apply(analyze).values

    mse_test = np.mean(np.square(test_true_carbon - predicted_test_carbon))
    r2 = result.rvalue**2

    info = {
        'mse': mse_test,
        'r2': r2,
        'slope': result.slope,
        'intercept': result.intercept,
        'fitting_df': fitting_df,
        'carbon_fitting_df': carbon_fitting_df,
        'si_fitting_df': si_fitting_df,
        'c_lines_df': c_lines_df,
        'si_lines_df': si_lines_df,
        'peak_areas': peak_areas,
        'training_peak_areas': training_peak_areas,
        'test_peak_areas': test_peak_areas,
        'true_carbon': true_carbon,
        'training_true_carbon': training_true_carbon,
        'test_true_carbon': test_true_carbon,
        'predicted_carbon': predicted_carbon,
        'predicted_training_carbon': predicted_training_carbon,
        'predicted_test_carbon': predicted_test_carbon,
        'method group': 'Peak Fitting',
    }
    return info


# %%
def PF_Exp_Falloff(training_df, test_df, training_exp_df, test_exp_df):
    """
    Perform peak fitting on the training and test data.
    """
    training_cols = training_df.columns
    test_cols = test_df.columns
    
    combined_exp_df = pd.concat([training_exp_df, test_exp_df], axis=0)
    combined_exp_df = combined_exp_df.loc[~combined_exp_df.index.duplicated(keep='first')]

    combined_df = pd.concat([training_df, test_df], axis=1)
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated(keep='first')]
    
    true_carbon = combined_exp_df['avg_carbon_portion'].values
    test_true_carbon = test_exp_df['avg_carbon_portion'].values
    training_true_carbon = training_exp_df['avg_carbon_portion'].values
    
    fitting_df, carbon_fitting_df, si_fitting_df, c_lines_df, si_lines_df = pfa.PeakFit(
        combined_df,
        bins=combined_df.index,
        c_window=[4.2, 4.7], 
        si_window=[4.2, 4.7],
        c_baseline='exp_falloff',
        si_baseline='linear',
        )
    
    fitting_df = fitting_df.set_index('label')
    training_fitting_df = fitting_df.loc[training_cols]
    test_fitting_df = fitting_df.loc[test_cols]
    
    peak_areas = fitting_df['Carbon Peak Area'].values
    training_peak_areas = training_fitting_df['Carbon Peak Area'].values
    test_peak_areas = test_fitting_df['Carbon Peak Area'].values
    
    result, analyze = activation_layer(training_peak_areas, training_true_carbon)
    
    predicted_carbon = fitting_df['Carbon Peak Area'].apply(analyze).values
    predicted_training_carbon = training_fitting_df['Carbon Peak Area'].apply(analyze).values
    predicted_test_carbon = test_fitting_df['Carbon Peak Area'].apply(analyze).values

    mse_test = np.mean(np.square(test_true_carbon - predicted_test_carbon))
    r2 = result.rvalue**2

    info = {
        'mse': mse_test,
        'r2': r2,
        'slope': result.slope,
        'intercept': result.intercept,
        'fitting_df': fitting_df,
        'carbon_fitting_df': carbon_fitting_df,
        'si_fitting_df': si_fitting_df,
        'c_lines_df': c_lines_df,
        'si_lines_df': si_lines_df,
        'peak_areas': peak_areas,
        'training_peak_areas': training_peak_areas,
        'test_peak_areas': test_peak_areas,
        'true_carbon': true_carbon,
        'training_true_carbon': training_true_carbon,
        'test_true_carbon': test_true_carbon,
        'predicted_carbon': predicted_carbon,
        'predicted_training_carbon': predicted_training_carbon,
        'predicted_test_carbon': predicted_test_carbon,
        'method group': 'Peak Fitting',
    }
    return info


# %% [markdown]
# ## 2. Component Analysis

# %%
def CAavg(training_df, test_df, training_exp_df, test_exp_df):
    """
    Perform component analysis on the training and test data.
    """
    training_cols = training_df.columns
    test_cols = test_df.columns
    
    combined_df = pd.concat([training_df, test_df], axis=1)
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
    combined_exp_df = pd.concat([training_exp_df, test_exp_df], axis=0)
    combined_exp_df = combined_exp_df.loc[~combined_exp_df.index.duplicated(keep='first')]

    true_carbon = combined_exp_df['avg_carbon_portion'].values
    test_true_carbon = test_exp_df['avg_carbon_portion'].values
    training_true_carbon = training_exp_df['avg_carbon_portion'].values
    
    edge_split_train_df, edge_split_train_exp_df = edge_split_train_avg(training_df, training_exp_df)

    _combined_df = pd.concat([combined_df, edge_split_train_df], axis=1)
    _combined_exp_df = pd.concat([combined_exp_df, edge_split_train_exp_df], axis=0)
    fitting_df, predicted_df = ca.Analyze(
        _combined_df,
        bins=combined_df.index,
        train_cols=list(edge_split_train_df.columns),
        true_c_concentrations=_combined_exp_df['avg_carbon_portion'].values,
        normalize=False,
        # initial_guess=np.array([1/len(edge_split_train_df.columns)] * len(edge_split_train_df.columns)),
        window = [4.2, 4.7]
    )
    predicted_df = predicted_df.loc[combined_df.columns]

    training_predicted_df = predicted_df.loc[training_cols]
    test_predicted_df = predicted_df.loc[test_cols]

    carbon_portions = predicted_df['Carbon Portion'].values
    training_carbon_portions = training_predicted_df['Carbon Portion'].values
    test_carbon_portions = test_predicted_df['Carbon Portion'].values

    result, analyze = activation_layer(training_carbon_portions, training_true_carbon)
    
    predicted_carbon = predicted_df['Carbon Portion'].apply(analyze).values
    predicted_training_carbon = training_predicted_df['Carbon Portion'].apply(analyze).values
    predicted_test_carbon = test_predicted_df['Carbon Portion'].apply(analyze).values

    mse_test = np.mean(np.square(test_true_carbon - predicted_test_carbon))
    r2 = result.rvalue**2

    info = {
        'mse': mse_test,
        'r2': r2,
        'slope': result.slope,
        'intercept': result.intercept,
        'fitting_df': fitting_df,
        'predicted_df': predicted_df,
        'carbon_portions': carbon_portions,
        'training_carbon_portions': training_carbon_portions,
        'test_carbon_portions': test_carbon_portions,
        'true_carbon': true_carbon,
        'training_true_carbon': training_true_carbon,
        'test_true_carbon': test_true_carbon,
        'predicted_carbon': predicted_carbon,
        'predicted_training_carbon': predicted_training_carbon,
        'predicted_test_carbon': predicted_test_carbon,
        'method group': 'Component Analysis',
    }
    return info


# %%
def CAelem(training_df, test_df, training_exp_df, test_exp_df, elem_df=elem_df, elem_exp_df=elem_exp_df):
    """
    Perform component analysis on the training and test data.
    """
    training_cols = training_df.columns
    test_cols = test_df.columns
    
    combined_df = pd.concat([training_df, test_df], axis=1)
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
    combined_exp_df = pd.concat([training_exp_df, test_exp_df], axis=0)
    combined_exp_df = combined_exp_df.loc[~combined_exp_df.index.duplicated(keep='first')]

    true_carbon = combined_exp_df['avg_carbon_portion'].values
    test_true_carbon = test_exp_df['avg_carbon_portion'].values
    training_true_carbon = training_exp_df['avg_carbon_portion'].values
    
    elem_df = elem_df.loc[combined_df.index]
    _combined_df = pd.concat([combined_df, elem_df], axis=1)
    _combined_exp_df = pd.concat([combined_exp_df, elem_exp_df], axis=0)
    fitting_df, predicted_df = ca.Analyze(
        _combined_df,
        bins=combined_df.index,
        train_cols=list(elem_exp_df.index),
        true_c_concentrations=_combined_exp_df['avg_carbon_portion'].values,
        normalize=False,
        window = [4.2, 4.7]
    )
    predicted_df = predicted_df.loc[combined_df.columns]

    training_predicted_df = predicted_df.loc[training_cols]
    test_predicted_df = predicted_df.loc[test_cols]

    carbon_portions = predicted_df['Carbon Portion'].values
    training_carbon_portions = training_predicted_df['Carbon Portion'].values
    test_carbon_portions = test_predicted_df['Carbon Portion'].values

    result, analyze = activation_layer(training_carbon_portions, training_true_carbon)
    
    predicted_carbon = predicted_df['Carbon Portion'].apply(analyze).values
    predicted_training_carbon = training_predicted_df['Carbon Portion'].apply(analyze).values
    predicted_test_carbon = test_predicted_df['Carbon Portion'].apply(analyze).values

    mse_test = np.mean(np.square(test_true_carbon - predicted_test_carbon))
    r2 = result.rvalue**2

    info = {
        'mse': mse_test,
        'r2': r2,
        'slope': result.slope,
        'intercept': result.intercept,
        'fitting_df': fitting_df,
        'predicted_df': predicted_df,
        'carbon_portions': carbon_portions,
        'training_carbon_portions': training_carbon_portions,
        'test_carbon_portions': test_carbon_portions,
        'true_carbon': true_carbon,
        'training_true_carbon': training_true_carbon,
        'test_true_carbon': test_true_carbon,
        'predicted_carbon': predicted_carbon,
        'predicted_training_carbon': predicted_training_carbon,
        'predicted_test_carbon': predicted_test_carbon,
        'method group': 'Component Analysis',
        
    }
    return info


# %%


# %% [markdown]
# ## 3. Singular Value Decomposition (SVD)

# %%
def SVD(training_df, test_df, training_exp_df, test_exp_df):
    """
    Perform peak fitting on the training and test data.
    """
    training_cols = training_df.columns
    test_cols = test_df.columns
    
    combined_df = pd.concat([training_df, test_df], axis=1)
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
    # print(combined_df.shape)
    combined_exp_df = pd.concat([training_exp_df, test_exp_df], axis=0)
    combined_exp_df = combined_exp_df.loc[~combined_exp_df.index.duplicated(keep='first')]
    # print(combined_exp_df.shape)

    true_carbon = combined_exp_df['avg_carbon_portion'].values
    test_true_carbon = test_exp_df['avg_carbon_portion'].values
    training_true_carbon = training_exp_df['avg_carbon_portion'].values
    
    fitting_df, decomposed_df, responses = svd.SVD(
        combined_df, 
        [4.2, 4.7], 
        [2, 10], 
        bins=combined_df.index
        )
    
    # fitting_df = fitting_df.set_index('label')
    training_fitting_df = fitting_df.loc[training_cols]
    test_fitting_df = fitting_df.loc[test_cols]
    
    peak_areas = fitting_df['Carbon Peak Area'].values
    training_peak_areas = training_fitting_df['Carbon Peak Area'].values
    test_peak_areas = test_fitting_df['Carbon Peak Area'].values

    result, analyze = activation_layer(training_peak_areas, training_true_carbon)
    
    predicted_carbon = fitting_df['Carbon Peak Area'].apply(analyze).values
    predicted_training_carbon = training_fitting_df['Carbon Peak Area'].apply(analyze).values
    predicted_test_carbon = test_fitting_df['Carbon Peak Area'].apply(analyze).values

    mse_test = np.mean(np.square(test_true_carbon - predicted_test_carbon))
    r2 = result.rvalue**2


    info = {
        'mse': mse_test,
        'r2': r2,
        'slope': result.slope,
        'intercept': result.intercept,
        'fitting_df': fitting_df,
        'decomposed_df': decomposed_df,
        'peak_areas': peak_areas,
        'training_peak_areas': training_peak_areas,
        'test_peak_areas': test_peak_areas,
        'true_carbon': true_carbon,
        'training_true_carbon': training_true_carbon,
        'test_true_carbon': test_true_carbon,
        'predicted_carbon': predicted_carbon,
        'predicted_training_carbon': predicted_training_carbon,
        'predicted_test_carbon': predicted_test_carbon,
        
    }
    return info


# %% [markdown]
# ## 4. Machine Learning

# %%
def ML(training_df, test_df, training_exp_df, test_exp_df):

    training_cols = training_df.columns
    test_cols = test_df.columns
    
    combined_df = pd.concat([training_df, test_df], axis=1)
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
    combined_exp_df = pd.concat([training_exp_df, test_exp_df], axis=0)
    combined_exp_df = combined_exp_df.loc[~combined_exp_df.index.duplicated(keep='first')]

    true_carbon = combined_exp_df['avg_carbon_portion'].values
    test_true_carbon = test_exp_df['avg_carbon_portion'].values
    training_true_carbon = training_exp_df['avg_carbon_portion'].values

    carbon_predictions_df, history = ml.kim2025(
        training_df=training_df,
        test_df=test_df,
        training_exp_df=training_exp_df,
        test_exp_df=test_exp_df
    )

    training_x = carbon_predictions_df['Carbon Portion'].loc[training_exp_df.index].values
    training_y = training_exp_df['avg_carbon_portion'].values

    # print("Training X:", training_x)
    # print("Training Y:", training_y)

    test_x = carbon_predictions_df['Carbon Portion'].loc[test_exp_df.index].values
    test_y = test_exp_df['avg_carbon_portion'].values

    result, analyze = activation_layer(training_x, training_y)

    predicted_carbon = carbon_predictions_df['Carbon Portion'].apply(analyze).values
    predicted_training_carbon = carbon_predictions_df['Carbon Portion'].loc[training_exp_df.index].apply(analyze).values
    predicted_test_carbon = carbon_predictions_df['Carbon Portion'].loc[test_exp_df.index].apply(analyze).values

    x = carbon_predictions_df['Carbon Portion'].values
    x_hat = carbon_predictions_df['Carbon Portion'].apply(analyze)
    x_hat = np.array(x_hat)
    mse_test = np.mean(np.square(test_y - analyze(test_x)))
    r2 = result.rvalue**2

    info = {
        'mse': mse_test,
        'r2': r2,
        'slope': result.slope,
        'intercept': result.intercept,
        'history': history,
        'training_x': training_x,
        'training_y': training_y,
        'test_x': test_x,
        'test_y': test_y,
        'x_hat': x_hat,
        'method group': 'Machine Learning',
        'true_carbon': true_carbon,
        'predicted_carbon': predicted_carbon,
        'predicted_training_carbon': predicted_training_carbon,
        'predicted_test_carbon': predicted_test_carbon
    }

    # training_predicted_df = carbon_predictions_df.loc[training_cols]
    # test_predicted_df = carbon_predictions_df.loc[test_cols]

    # carbon_portions = carbon_predictions_df['Carbon Portion'].values
    # training_carbon_portions = training_predicted_df['Carbon Portion'].values
    # test_carbon_portions = test_predicted_df['Carbon Portion'].values

    # result = linregress(training_carbon_portions, training_true_carbon)

    # analyze = lambda x: result.slope * x + result.intercept
    
    # predicted_carbon = carbon_predictions_df['Carbon Portion'].apply(analyze).values
    # predicted_training_carbon = training_predicted_df['Carbon Portion'].apply(analyze).values
    # predicted_test_carbon = test_predicted_df['Carbon Portion'].apply(analyze).values

    # mse_test = np.mean(np.square(test_true_carbon - predicted_test_carbon))
    # r2 = result.rvalue**2

    # info = {
    #     'mse': mse_test,
    #     'r2': r2,
    #     'slope': result.slope,
    #     'intercept': result.intercept,
    #     'carbon_portions': carbon_portions,
    #     'training_carbon_portions': training_carbon_portions,
    #     'test_carbon_portions': test_carbon_portions,
    #     'true_carbon': true_carbon,
    #     'training_true_carbon': training_true_carbon,
    #     'test_true_carbon': test_true_carbon,
    #     'predicted_carbon': predicted_carbon,
    #     'predicted_training_carbon': predicted_training_carbon,
    #     'predicted_test_carbon': predicted_test_carbon,
        
    # }
    return info

# %%
def Filtered_ML(training_df, test_df, training_exp_df, test_exp_df):

    spectral_filter = [1, 5]
    training_cols = training_df.columns
    test_cols = test_df.columns


    training_df = training_df.loc[(training_df.index <= spectral_filter[1]) & (training_df.index >= spectral_filter[0])]
    test_df = test_df.loc[(test_df.index <= spectral_filter[1]) & (test_df.index >= spectral_filter[0])]
    print("shapes: ", training_df.shape, test_df.shape)
    combined_df = pd.concat([training_df, test_df], axis=1)
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
    combined_exp_df = pd.concat([training_exp_df, test_exp_df], axis=0)
    combined_exp_df = combined_exp_df.loc[~combined_exp_df.index.duplicated(keep='first')]

    true_carbon = combined_exp_df['avg_carbon_portion'].values
    test_true_carbon = test_exp_df['avg_carbon_portion'].values
    training_true_carbon = training_exp_df['avg_carbon_portion'].values

    carbon_predictions_df, history = ml.kim2025(
        training_df=training_df,
        test_df=test_df,
        training_exp_df=training_exp_df,
        test_exp_df=test_exp_df
    )

    training_x = carbon_predictions_df['Carbon Portion'].loc[training_exp_df.index].values
    training_y = training_exp_df['avg_carbon_portion'].values

    # print("Training X:", training_x)
    # print("Training Y:", training_y)

    test_x = carbon_predictions_df['Carbon Portion'].loc[test_exp_df.index].values
    test_y = test_exp_df['avg_carbon_portion'].values

    result, analyze = activation_layer(training_x, training_y)

    x = carbon_predictions_df['Carbon Portion'].values
    x_hat = carbon_predictions_df['Carbon Portion'].apply(analyze)
    x_hat = np.array(x_hat)
    mse_test = np.mean(np.square(test_y - analyze(test_x)))
    r2 = result.rvalue**2

    info = {
        'mse': mse_test,
        'r2': r2,
        'slope': result.slope,
        'intercept': result.intercept,
        'history': history,
        'training_x': training_x,
        'training_y': training_y,
        'test_x': test_x,
        'test_y': test_y,
        'x_hat': x_hat,
    }

    # training_predicted_df = carbon_predictions_df.loc[training_cols]
    # test_predicted_df = carbon_predictions_df.loc[test_cols]

    # carbon_portions = carbon_predictions_df['Carbon Portion'].values
    # training_carbon_portions = training_predicted_df['Carbon Portion'].values
    # test_carbon_portions = test_predicted_df['Carbon Portion'].values

    # result = linregress(training_carbon_portions, training_true_carbon)

    # analyze = lambda x: result.slope * x + result.intercept
    
    # predicted_carbon = carbon_predictions_df['Carbon Portion'].apply(analyze).values
    # predicted_training_carbon = training_predicted_df['Carbon Portion'].apply(analyze).values
    # predicted_test_carbon = test_predicted_df['Carbon Portion'].apply(analyze).values

    # mse_test = np.mean(np.square(test_true_carbon - predicted_test_carbon))
    # r2 = result.rvalue**2

    # info = {
    #     'mse': mse_test,
    #     'r2': r2,
    #     'slope': result.slope,
    #     'intercept': result.intercept,
    #     'carbon_portions': carbon_portions,
    #     'training_carbon_portions': training_carbon_portions,
    #     'test_carbon_portions': test_carbon_portions,
    #     'true_carbon': true_carbon,
    #     'training_true_carbon': training_true_carbon,
    #     'test_true_carbon': test_true_carbon,
    #     'predicted_carbon': predicted_carbon,
    #     'predicted_training_carbon': predicted_training_carbon,
    #     'predicted_test_carbon': predicted_test_carbon,
        
    # }
    return info
# %% [markdown]
# ## Training and Testing

# %%
functions = {
    "Baseline and Peak Fitting - linear Baseline": PF_Linear,
    "Baseline and Peak Fitting - Exponential Falloff Baseline": PF_Exp_Falloff,
    "Component Analysis - Average Training": CAavg,
    "Component Analysis - Elemental Maps": CAelem,
    "Convex Optimization": SVD,
    "Machine Learning": ML,
    "Filtered Machine Learning": Filtered_ML,
}

# %%
save = []
for dataset_name, dataset in datasets.items():
    train_df = dataset['train_df']
    train_exp_df = dataset['train_exp_df']
    test_df = dataset['test_df']
    test_exp_df = dataset['test_exp_df']
    
    for level_name, level in Carbon_Levels.items():
        min_carbon = level['min']
        max_carbon = level['max']
        
        train_filtered_df, train_filtered_exp_df = lev_filter(train_df, train_exp_df, min_carbon, max_carbon)
        test_filtered_df, test_filtered_exp_df = lev_filter(test_df, test_exp_df, min_carbon, max_carbon)
        
        if dataset['train_filter'] == 'edge':
        
            train_filtered_df, train_filtered_exp_df = edge_split_train(train_filtered_df, train_filtered_exp_df)
            
        for function_name, function in functions.items():
            try:
                info = function(train_filtered_df, test_filtered_df, train_filtered_exp_df, test_filtered_exp_df)
                info['datasets used'] = dataset_name
                info['carbon level'] = level_name
                info['method'] = function_name
                save.append(info)
            except Exception as e:
                print(f"Error occurred while processing {function_name} for {dataset_name} at {level_name}: {e}")
                # throw an error instead of continuing
                raise e
            

# %%
output = pd.DataFrame(save)#[['datasets used', 'carbon level', 'method', 'mse']]
output['id'] = output.index
output

# pickle results
output.to_pickle('analysis_results.pkl')

# # %% [markdown]
# # # Plotting

# # %%
# import seaborn as sns
# sns.set_context('paper', font_scale = 1)

# # %%
# figure_folder = '../../Figures/Analysis/'

# # %% [markdown]
# # ## Peak Fitting

# # %%
# output

# # %%
# pfdf = output[output['method group']=="Peak Fitting"].sort_values(by='mse', ascending=True)
# print(tabulate(
#     pfdf[['method', 'carbon level', 'datasets used', 'mse']], 
#     headers='keys',
#     showindex=False,
#     tablefmt='github',
#         ))

# # %%
# _df = output[output['method']=="Peak Fitting - Exponential Falloff Baseline"].sort_values(by='mse')['c_lines_df'].iloc[0]
# pfdf = output[output['method']=="Peak Fitting - Exponential Falloff Baseline"].sort_values(by='mse', ascending=True)

# # %%
# # where carbon level is 'Agricultural' and datasets used is 'Material Mixes'
# explode_vals_df = pfdf[(pfdf['carbon level'] == 'Agricultural') & (pfdf['datasets used'] == 'Material Mixes')].copy()
# explode_vals_df = explode_vals_df.explode([
#     'true_carbon', 
#     'predicted_carbon',    
#     'peak_areas',
#     ], ignore_index=True)

# # %%
# plt.figure(figsize=(4, 3.54330709))
# sns.scatterplot(data=explode_vals_df, x='true_carbon', y='predicted_carbon', alpha=1, label='True vs Predicted Carbon')

# plt.plot([0, .06], [0, .06], color='gray', linestyle='--', linewidth=2, label='Ideal Fit')
# plt.gca().set_aspect('equal')
# # plt.xlim(-0.01, 0.07)
# # plt.ylim(-0.01, 0.07)
# plt.legend(loc='upper left', fontsize=6)
# plt.title('Peak Fitting - Exponential Falloff Baseline\nAgricultural Carbon Levels\nSimulated Material Mixes')
# plt.xlabel('True Carbon Portion')
# plt.ylabel('Predicted Carbon Portion')
# plt.savefig(figure_folder + 'PF_Exponential_Falloff_Agricultural_Carbon_Levels.jpg', dpi=100, bbox_inches='tight')
# plt.show()

# # %%
# _df.columns
# # filter to if the column contains 'Feldspar' or '7x7x7'
# feldspar_columns = _df.filter(like='Feldspar').columns
# seven_by_seven_columns = _df.filter(like='7x7x7').columns
# both_columns = feldspar_columns.intersection(seven_by_seven_columns)

# # %%
# min_carbon_example_fitting_cols = ['7x7x7_Feldspar_001021 baseline', '7x7x7_Feldspar_001021 peak','7x7x7_Feldspar_001021 true',]
# max_carbon_example_fitting_cols = ['7x7x7_C_0600_Feldspar_Fill_003142 baseline', '7x7x7_C_0600_Feldspar_Fill_003142 peak', '7x7x7_C_0600_Feldspar_Fill_003142 true']

# # %%
# # plot every column in _df
# plt.figure()
# # for col in min_carbon_example_fitting_cols:
# plt.plot(_df.bins, _df['7x7x7_C_0600_Feldspar_Fill_003142 true'], label='6% Carbon in Feldspar', color='tab:green', linewidth=3, alpha=1)
# plt.plot(_df.bins, _df['7x7x7_Feldspar_001021 true'], label='0% Carbon in Feldspar', color='tab:blue', linewidth=3, alpha=1)

# plt.plot(_df.bins, _df['7x7x7_Feldspar_001021 peak'], label='Peak', color='red', linestyle='dotted', alpha=1, linewidth=5)
# plt.plot(_df.bins, _df['7x7x7_Feldspar_001021 baseline'], label='Baseline', color='black', linestyle='--', alpha=1)
# plt.plot(_df.bins, _df['7x7x7_C_0600_Feldspar_Fill_003142 peak'], color='red', linestyle='dotted', alpha=1, linewidth=5)
# plt.plot(_df.bins, _df['7x7x7_C_0600_Feldspar_Fill_003142 baseline'], color='black', linestyle='--', alpha=1)
# plt.legend()
# plt.xlabel('Bins')
# plt.ylabel('Intensity')
# plt.title('Peak Fitting for Feldspar with Varying Carbon Levels\nExponential Falloff Baseline')
# plt.savefig(figure_folder+'peak_fitting_feldspar.png', dpi=50, bbox_inches='tight')
# plt.show()

# # %% [markdown]
# # ## Component Analysis

# # %%
# output.method.unique()

# # %%
# output['method group'].unique()

# # %%
# # _df = output[output['method']=='Peak Fitting - linear Baseline'].sort_values(by='mse')['c_lines_df'].iloc[0]
# pfdf = output[output['method group']=='Component Analysis'].sort_values(by='mse', ascending=True)
# print(tabulate(
#     pfdf[['method', 'carbon level', 'datasets used', 'mse']], 
#     headers='keys',
#     showindex=False,
#     tablefmt='github',
#         ))

# # %%

# _df = pfdf
# _df = _df[_df['carbon level'] == 'Agricultural']
# _df = _df[_df['datasets used'].isin(['Material Mixes', 'Feldspar'])]
# explode_test_vals_df = _df.explode([
#     'true_carbon', 
#     'predicted_carbon',
#     ], ignore_index=True)
# # explode_test_vals_df = explode_test_vals_df.sort_values(by='datasets used', ascending=False)

# plt.figure()
# sns.scatterplot(data=explode_test_vals_df, x='true_carbon', y='predicted_carbon', style='datasets used', hue='method', alpha=.5)

# plt.plot([0, .06], [0, .06], color='gray', linestyle='--')
# plt.gca().set_aspect('equal')
# plt.xlim(-0.01, 0.07)
# plt.ylim(-0.01, 0.07)

# plt.legend(loc='upper left', fontsize=6)
# plt.title('Component Analysis - Agricultural Carbon Levels\nSimulated Material Mixes and Feldspar')
# plt.xlabel('True Carbon Portion')
# plt.ylabel('Predicted Carbon Portion')
# plt.savefig(figure_folder + 'CA_Agricultural_Carbon_Levels.jpg', dpi=100, bbox_inches='tight')
# plt.show()

# # %%
# _df = output[output['datasets used']=='Feldspar']
# _df = _df[_df['method']=='Component Analysis - Average Training']
# explode_test_vals_df = _df.explode([
#     'true_carbon', 
#     'predicted_carbon',
#     ], ignore_index=True)

# plt.figure()
# sns.scatterplot(data=explode_test_vals_df, x='true_carbon', y='predicted_carbon', style='datasets used', hue='carbon level', alpha=1)

# plt.plot([0, .06], [0, .06], color='gray', linestyle='--')
# plt.gca().set_aspect('equal')
# plt.xlim(0, 0.06)
# plt.ylim(0, 0.06)
# plt.show()

# # %%
# combined = (df['7x7x7_Feldspar_001021']*_df.iloc[0]['fitting_df'].loc['7x7x7_C_0300_Feldspar_Fill_003137']['average_min_carbon'])+(df['7x7x7_C_0600_Feldspar_Fill_003142']*_df.iloc[0]['fitting_df'].loc['7x7x7_C_0600_Feldspar_Fill_003142']['average_max_carbon'])

# plt.figure()
# plt.plot(bins, df['7x7x7_C_0600_Feldspar_Fill_003142'], label='6% Carbon in Feldspar', color='tab:green', linewidth=3, alpha=0.5)
# plt.plot(bins, df['7x7x7_C_0300_Feldspar_Fill_003137'], label='3% Carbon in Feldspar', color='tab:orange', linewidth=3, alpha=0.5)
# plt.plot(bins, df['7x7x7_Feldspar_001021'], label='0% Carbon in Feldspar', color='tab:blue', linewidth=3, alpha=0.5)
# plt.plot(bins, combined, label='linear combination', color='tab:red', linewidth=3, linestyle='dotted')
# plt.xlabel('Bins')
# plt.ylabel('Probability')
# plt.title('Linear Combination of Feldspar Spectra found by component fitting')
# plt.legend()
# plt.xlim(4.2, 4.7)
# plt.ylim(8e-7, 2e-6)
# plt.savefig(figure_folder+'linear_combination_feldspar.png', dpi=50, bbox_inches='tight')
# plt.show()

# # %% [markdown]
# # ## Singular Value Decomposition (SVD)

# # %%
# pfdf = output[output['method']=='SVD'].sort_values(by='mse', ascending=True)
# print(tabulate(
#     pfdf[['carbon level', 'datasets used', 'mse']], 
#     headers='keys',
#     showindex=False,
#     tablefmt='github',
#         ))

# # %%


# # %%
# example_decomposed_df = output[output.method == 'SVD'].iloc[0]['decomposed_df'].T

# # %%
# example_decomposed_df

# # %%
# _df = _df[_df['method']=='Component Analysis - Average Training']

# # %%
# plt.figure()
# plt.plot(bins, df['7x7x7_C_0600_Feldspar_Fill_003142'], label='6% Carbon in Feldspar', color='tab:green', linewidth=3, alpha=0.5)
# plt.stem(example_decomposed_df.index, example_decomposed_df['7x7x7_C_0600_Feldspar_Fill_003142'],'tab:green')
# plt.plot(bins, df['7x7x7_C_0300_Feldspar_Fill_003137'], label='3% Carbon in Feldspar', color='tab:orange', linewidth=3, alpha=0.5)
# plt.stem(example_decomposed_df.index, example_decomposed_df['7x7x7_C_0300_Feldspar_Fill_003137'], 'tab:orange')
# plt.plot(bins, df['7x7x7_Feldspar_001021'], label='0% Carbon in Feldspar', color='tab:blue', linewidth=3, alpha=0.5)
# plt.stem(example_decomposed_df.index, example_decomposed_df['7x7x7_Feldspar_001021'], 'tab:blue')
# # make empty lines for legend
# # plt.plot([], [], label='6% Carbon in Feldspar', color='tab:green', linewidth=3, alpha=0.5)
# # plt.plot([], [], label='3% Carbon in Feldspar', color='tab:orange', linewidth=3, alpha=0.5)
# # plt.plot([], [], label='0% Carbon in Feldspar', color='tab:blue', linewidth=3, alpha=0.5)
# plt.scatter([], [], label='SVD Decomposition', color='black', linewidth=3, alpha=1)
# plt.xlabel('Bins')
# plt.ylabel('Probability')
# plt.title('Decomposed Feldspar Spectra from SVD')
# plt.legend()
# plt.yscale('log')
# # plt.xlim(4.2, 4.7)
# plt.xlim(4, 5)
# plt.ylim(5e-7, 6e-5)
# plt.savefig(figure_folder+'decomposed_feldspar_svd.jpg', dpi=50, bbox_inches='tight')
# # save as eps
# # plt.savefig(figure_folder+'decomposed_feldspar_svd.eps', dpi=50, bbox_inches='tight')
# plt.show()

# # %%


# # %%


# # %% [markdown]
# # ## Deep Learning

# # %% [markdown]
# # 

# # %%
# output.method.unique()

# # %%
# pfdf = output[output['method']=='Machine Learning'].sort_values(by='mse', ascending=True)
# print(tabulate(
#     pfdf[['carbon level', 'datasets used', 'mse']], 
#     headers='keys',
#     showindex=False,
#     tablefmt='github',
#         ))

# # %% [markdown]
# # ## All

# # %%
# pfdf = output.sort_values(by='mse', ascending=True)
# print(tabulate(
#     pfdf[['method', 'carbon level', 'datasets used', 'mse']], 
#     headers='keys',
#     showindex=False,
#     tablefmt='github',
#         ))

# # %% [markdown]
# # 

# # %%
# # make this table:
# # |Carbon Level | Component Fitting MSE | Peak Fitting MSE | SVD MSE | Deep Learning MSE |
# # |-------------|-----------------------|------------------|---------|-------------------|
# # | Low         | 0.0x                  | 0.0x             | 0.0x    | 0.0x              |
# # | High        | 0.0x                  | 0.0x             | 0.0x    | 0.0x              |
# carbon_levels = output['carbon level'].unique()
# methods = output['method'].unique()
# results_table = pd.DataFrame(columns=['Carbon Level'] + list(methods))

# for level in carbon_levels:
#     row = {'Carbon Level': level}
#     for method in methods:
#         mse = output[(output['carbon level'] == level) & (output['method'] == method)]['mse']
#         if not mse.empty:
#             row[method] = mse.values[0]
#         else:
#             row[method] = 'N/A'
#     results_table = pd.concat([results_table, pd.DataFrame([row])], ignore_index=True)
# print(tabulate(
#     results_table, 
#     headers='keys',
#     showindex=False,
#     tablefmt='github',
#         ))

# # %%
# # do the same for the datasets used
# datasets_used = output['datasets used'].unique()
# results_table_datasets = pd.DataFrame(columns=['Datasets Used'] + list(methods))
# for dataset in datasets_used:
#     row = {'Datasets Used': dataset}
#     for method in methods:
#         mse = output[(output['datasets used'] == dataset) & (output['method'] == method)]['mse']
#         if not mse.empty:
#             row[method] = mse.values[0]
#         else:
#             row[method] = 'N/A'
#     results_table_datasets = pd.concat([results_table_datasets, pd.DataFrame([row])], ignore_index=True)
# print(tabulate(
#     results_table_datasets, 
#     headers='keys',
#     showindex=False,
#     tablefmt='github',
#         ))


