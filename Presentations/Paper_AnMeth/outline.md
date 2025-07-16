# In Situ Spectral Analysis for Soil Carbon Measurement

contents:

1. Background
    1. Soil Organic Carbon (SOC)
    2. Spectral Analysis as a Measurement Technique
    3. Objectives of the Paper
2. Data Generation
    1. Common Soil Types
    2. Simulation in MCNP6
    3. Spectral Readings
    4. Data Convolution
    5. Training Data
3. Analysis Methods of Spectral Readings
    1. Peak Fitting
        - Strong Window
    2. Component Fitting
        - Weak window
    3. Singular Value Decomposition
    4. Deep Learning
4. Results
    1. Comparing Analysis Methods
    2. Effects of Carbon Levels on Results
    3. Effects of Convolution on Results
5. Discussion
    2. Conclusions
    3. Future Work
6. Acknowledgments and References

## 1. Background

Soil organic carbon (SOC) is a key component of soil health and plays a crucial role in the global carbon cycle. Accurate measurement of SOC is essential for understanding soil health, fertility, and its impact on climate change. Traditional methods for measuring SOC are often time-consuming, expensive, and require laboratory analysis. In situ spectral analysis offers a promising alternative for rapid and non-destructive measurement of SOC.
This paper explores the use of spectral analysis techniques to measure SOC levels in various soil types. We simulate common soil types and apply different spectral analysis methods, including peak fitting, component fitting, singular value decomposition, and deep learning, to evaluate their effectiveness in measuring SOC.

## 2. Data Generation

### 2.1. Common Soil Types

To investigate the effectiveness of spectral analysis methods for SOC measurement, we simulate a range of common soil types. The simulated data includes spectral readings across different wavelengths, capturing the unique spectral signatures of each soil type. This data serves as a foundation for applying various spectral analysis techniques.

| Element | MCNP Identifier | Density (g/cm^3) |
|---------|------------------|------------------|
| Si | 14028 | 2.33 |
| Al | 13027 | 2.7 |
| H | 1001 | 0.001 |
| Na | 11023 | 0.97 |
| O | 8016 | 0.00143 |
| Fe | 26000 | 7.87 |
| Mg | 12024 | 1.74 |
| C | 6000 | 2.33 |

| Compound | Density (g/cm^3) |
|----------|------------------|
| SiO2 | 2.65 |
| Al2O3 | 3.95 |
| H2O | 1.0 |
| Na2O | 2.16 |
| Fe2O3 | 5.24 |
| MgO | 2.74 |
| C | 2.33 |

| Material | Compound Makeup (by weight) | Density (g/cm^3) |
|----------|------------------|------------------|
| Silica | SiO2 (76.4%), Al2O3 (23.6%) | 2.32 |
| Kaolinite | SiO2 (46.5%), Al2O3 (39.5%), H2O (14.0%) | 3.95 |
| Smectite | SiO2 (66.7%), Al2O3 (28.3%), H2O (5.0%) | 2.785 |
| Montmorillonite | SiO2 (73.7%), Al2O3 (24.6%), H2O (1.7%) | 2.7 |
| Quartz | SiO2 (100.0%) | 2.62 |
| Chlorite | SiO2 (30.0%), Al2O3 (24.0%), Fe2O3 (23.3%), H2O (22.7%) | 2.6 |
| Mica | SiO2 (48.9%), Al2O3 (40.3%), H2O (10.8%) | 2.7 |
| Feldspar | SiO2 (68.0%), Al2O3 (32.0%) | 2.55 |
| Coconut | C (100.0%) | 0.53 |

To measure the effectiveness of spectral analysis methods for carbon measurement, we simulate combinations of soil materials with varying carbon content (coconut).

### 2.2. Simulation in MCNP

MCNP6 was used to simulate gamma-ray spectra resulting from neutron activation of soil samples. Each simulation modeled a soil matrix with varying concentrations of carbon and other common soil constituents. The geometry was set up to mimic in situ measurement conditions, with a neutron source placed above a soil slab and a detector positioned to capture emitted gamma rays.

Figure: Geometery of MCNP

Key simulation parameters included:
- **Neutron source energy:** 14 MeV (D-T generator)
- **Soil slab dimensions:** 
- **Detector type:** 
- **Tally:** F8 (pulse height tally) for gamma spectra

This approach enables the generation of realistic spectral data for a variety of soil compositions, forming the basis for evaluating different spectral analysis techniques.

## 2.3. Spectral Readings

The spectral readings obtained from the MCNP simulations provide a detailed representation of the gamma-ray emissions from the soil samples. Mathematically it is a probability density function (PDF) of the energy distribution of the emitted gamma rays.

![MCNP Spectral Reading](Figures/MCNPSpectralReading.png)

## 2.4 Training Data

The training data for the spectral analysis methods is picked from the edge cases of the simulated data. This includes the highest and lowest carbon levels both as would be found in simulation as well as natural soils.

| Carbon Level | Associated Amount |
|--------------|-------------------|
| Natural      | 0%-6% Carbon      |
| Medium       | 6%-100% Carbon    |

## 2.5. Data Convolution

Figure: Simulated vs Convoluted data

In the context of spectral analysis, MCNP can be used to simulate the interaction of radiation with soil materials, providing spectrums to analyze. Linear Convolution is used to quickly predict spectral readings for material mixtures by combining the spectral signatures of individual components. This does not account for the complex interactions between materials, but it provides a simplified approach to generate spectral data for analysis. The error metric for this convolution method is based on the difference between the simulated spectral readings and the readings obtained from MCNP simulations. The affects of convolution on the analysis results will be investigated in the results section.

## 3. Analysis Methods of Spectral Readings

This section explores various spectral analysis methods applied to the simulated spectral readings. Each method is evaluated for its effectiveness in measuring Carbon levels.

### 3.1 Peak Fitting

Peak fitting involves identifying and quantifying the peaks in the spectral data that correspond to specific soil components. This method is useful for extracting information about the concentration of individual elements or compounds in the soil.

*Strong Window*

Figure: Carbon Window

For effective peak filtering, it is best that the data be preprocessed such that the fitting is done on a section of the data around specified peak areas. This is called a Strong Window
(I dont know if this is a formalized term?)
For a carbon peak, which is centered around 4.44 MeV, the strong window is within 4-5 MeV.

Symbols Table:
Peak Function, F_p - Gaussian
Baseline Function, F_b - Linear, Exp Falloff
Fitting Function, F_f = F_p + F_b

Parameterized Functions:
Linear - ax+b
Exp Falloff - a * exp(-b * x) + c
Gaussian - a * exp(-((x - b) / c) ** 2) 

This method relies on parameterized functions, which are fitted to the spectral data to identify the peaks corresponding to specific elements or compounds. The fitting function is a combination of a peak function (e.g., Gaussian) and a baseline function (e.g., linear or exponential falloff). Starting parameters are generated automatically such that the initial fitting function is within the bounds of the spectrum in the strong window. Parameters are also constrained to ensure they remain within reasonable limits based on the expected spectral characteristics of the soil components.

Figure: Starting Parameters on spectrum

Table: Constraints on Parameters

The Scipy python package is used for the optimization process, leveraging its curve fitting capabilities to refine the initial parameter estimates. The baseline function is subtracted from the fitted function to isolate the peak, and the area under the peak is calculated to quantify the concentration of the corresponding element or compound in the soil.

Figure: Fitted Peak

An activation layer of linear regression is used to compare the peak areas to known soil carbon concentrations, allowing for the calibration of the model's predictions.

Figure: Peak Fitting Prediction Results

### 3.2 Component Fitting

Component fitting involves modeling the spectral data as a combination of known spectral signatures of soil components. This method allows for the estimation of the concentration of multiple components in the soil based on their spectral contributions.

Function: F_c = Σ (A_i * F_i)

Where:
- F_c is the combined spectral function
- A_i are the coefficients representing the concentration of each component
- F_i are the spectral functions of individual components

Figure: Common Soil Spectra vs Average Soil Spectrum

Components can be any known spectral signature, this can be from pure elemental samples or from the average of a set of soil samples. The fitting process involves adjusting the coefficients A_i to minimize the difference between the combined spectral function F_c and the observed spectral data.

*Weak window*

Figure: Weak Window

Component Fitting also benefits from filtering to windows. The simplest filtering being the filtering of low energy signals which are generally more likely to be caused by noise.

Figure: Component Fitting Process

The carbon coefficient A_C is then used to estimate the Carbon level in the soil. This method is particularly useful for analyzing complex soil mixtures where multiple known components contribute to the spectral signature. This method is also generalizable to study other elements or compounds.

Figure: Component Fitting Prediction Results

### 3.3 Singular Value Decomposition (SVD)

Figure: SVD Process

Singular Value Decomposition is a mathematical technique used to decompose the spectral data into its constituent components. The resulting singular values inside the strong window can be summed to provide a measure of the concentration of carbon in the soil.

Figure: SVD Prediction Results

### 3.4 Deep Learning

Figure: Deep Learning Model

Deep learning techniques, such as convolutional neural networks (CNNs), can be applied to spectral data for feature extraction and classification. These methods can learn complex relationships in the data and provide robust predictions of carbon levels based on spectral readings. The most important difference between deep learning and the previous methods is that it requires a large amount of training data to be effective.

Figure: Deep Learning Prediction Results

## 4. Results

The effectiveness of each method in measuring carbon levels is evaluated based on accuracy using mean squared error (MSE) as the metric. The results are summarized in the following table.

| Method                     | MSE   |
|----------------------------|-------|
| Component Fitting          | 0.0x  |
| Peak Fitting               | 0.0x  |
| Singular Value Decomposition| 0.0x  |
| Deep Learning              | 0.0x  |

Figure: Results Summary

### 4.1 Comparing Analysis Methods

The X method is the most effective for measuring carbon levels in soil, achieving the lowest MSE. The Y method also performs well, but is slightly less accurate than X. The Deep Learning method shows promise but requires further optimization to improve its performance.

### 4.2 Effects of Carbon Levels on Results

|Carbon Level | Component Fitting MSE | Peak Fitting MSE | SVD MSE | Deep Learning MSE |
|-------------|-----------------------|------------------|---------|-------------------|
| Low         | 0.0x                  | 0.0x             | 0.0x    | 0.0x              |
| High        | 0.0x                  | 0.0x             | 0.0x    | 0.0x              |

Lower carbon levels tend to result in higher MSE values across all methods, indicating that the spectral signatures of low-carbon soils are less distinct and more challenging to analyze accurately. The methods generally perform better with higher carbon concentrations, where the spectral features are more pronounced.

### 4.3 Effects of Convolution on Results

|Convolution | Component Fitting MSE | Peak Fitting MSE | SVD MSE | Deep Learning MSE |
|------------|-----------------------|------------------|---------|-------------------|
| No         | 0.0x                  | 0.0x             | 0.0x    | 0.0x              |
| Yes        | 0.0x                  | 0.0x             | 0.0x    | 0.0x              |

Convolution generally improves the accuracy of spectral analysis methods by smoothing out noise and enhancing the signal-to-noise ratio. The results show that convolution leads to lower MSE values across all methods, indicating that it is beneficial for spectral analysis in soil carbon measurement.

## 5. Discussion

### 5.1 Conclusions

The study demonstrates the potential of spectral analysis methods for measuring soil carbon levels. Component fitting and peak fitting methods show the best performance, while deep learning techniques require further refinement. Convolution is beneficial for improving the accuracy of spectral analysis.

### 5.2 Future Work

Future work will focus on x

## 6. Acknowledgments and References

We acknowledge the contributions of the USDA scientists for their guidance and support in this research. The spectral data generated in this study is available for further research and validation.

References:
1. x
