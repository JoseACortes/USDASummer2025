# In Situ Spectral Analysis for Soil Carbon Measurement

contents:

1. Background
    1. Soil Organic Carbon (SOC)
    2. Spectral Analysis as a Measurement Technique
    3. Objectives of the Paper
2. Simulating Common Soil Types - Data
3. Data Convolution
4. Analysis Methods of Spectral Readings
    1. Peak Fitting
        - Strong Window
    2. Component Fitting
        - Weak window
    3. Singular Value Decomposition
    4. Activation Layer
    5. Deep Learning
5. Results
    1. Comparing Analysis Methods
    2. Effects of Carbon Level on Results
    3. Effects of Convolution on Results
6. Discussion
    2. Conclusions
    3. Future Work
7. Acknowledgments and References

## 1. Background

Soil organic carbon (SOC) is a key component of soil health and plays a crucial role in the global carbon cycle. Accurate measurement of SOC is essential for understanding soil health, fertility, and its impact on climate change. Traditional methods for measuring SOC are often time-consuming, expensive, and require laboratory analysis. In situ spectral analysis offers a promising alternative for rapid and non-destructive measurement of SOC.
This paper explores the use of spectral analysis techniques to measure SOC levels in various soil types. We simulate common soil types and apply different spectral analysis methods, including peak fitting, component fitting, singular value decomposition, and deep learning, to evaluate their effectiveness in measuring SOC.

## 2. Simulating Common Soil Types - Data

To investigate the effectiveness of spectral analysis methods for SOC measurement, we simulate a range of common soil types. The simulated data includes spectral readings across different wavelengths, capturing the unique spectral signatures of each soil type. This data serves as a foundation for applying various spectral analysis techniques.

elem_names = ['Si', 'Al', 'H', 'Na', 'O', 'Fe', 'Mg', 'C']
compound_names = ['SiO2', 'Al2O3', 'H2O', 'Na2O', 'Fe2O3', 'MgO', 'C']
material_names = ['Silica', 'Kaolinite', 'Smectite', 'Montmorillonite', 'Quartz', 'Chlorite', 'Mica', 'Feldspar', 'Coconut']

## 3. Data Convolution

MCNP (Monte Carlo N-Particle Transport Code) is a general-purpose code for simulating nuclear processes, including neutron, photon, and electron transport. In the context of spectral analysis, MCNP can be used to simulate the interaction of radiation with soil materials, providing spectrums to analyze. Linear Convolution is used to quickly simulate spectral readings for material mixtures by combining the spectral signatures of individual components. This does not account for the complex interactions between materials, but it provides a simplified approach to generate spectral data for analysis. The error metric for this convolution method is based on the difference between the simulated spectral readings and the actual readings obtained from MCNP simulations. The affects of convolution on the analysis results will be investigated in the results section.

## 4. Analysis Methods of Spectral Readings

This section explores various spectral analysis methods applied to the simulated spectral readings. Each method is evaluated for its effectiveness in measuring SOC levels.

### 4.1 Peak Fitting

Peak fitting involves identifying and quantifying the peaks in the spectral data that correspond to specific soil components. This method is useful for extracting information about the concentration of individual elements or compounds in the soil.

*Strong Window*
For effective peak filtering, it is best that the data be preprocessed such that the fitting is done on a section of the data around specified peak areas. This is called a Strong Window
(I dont know if this is a formalized term?)
For a carbon peak, which is centered around 4.44 MeV, the strong window is within 4-5 MeV.

### 4.2 Component Fitting

Component fitting involves modeling the spectral data as a combination of known spectral signatures of soil components. This method allows for the estimation of the concentration of multiple components in the soil based on their spectral contributions.

*Weak window*
Component Fitting also benifits from filtering to windows. The simplest filtering being the filtering of low energy signals which are generally more likely to be caused by noise.

### 4.3 Singular Value Decomposition (SVD)

Singular Value Decomposition is a mathematical technique used to decompose the spectral data into its constituent components. SVD can help identify patterns and relationships in the spectral data, making it useful for analyzing complex soil mixtures.

### 4.4 Deep Learning

Deep learning techniques, such as convolutional neural networks (CNNs), can be applied to spectral data for feature extraction and classification. These methods can learn complex relationships in the data and provide robust predictions of SOC levels based on spectral readings.
