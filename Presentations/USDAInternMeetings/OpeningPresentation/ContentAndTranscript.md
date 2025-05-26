# Intro/Title Slide

Title: USDA Summer Intern Meeting  
Subtitle: Opening Presentation  
Author: Jose Andres Cortes^1  
CoAuthors: Dr. Korzeniowski^1, Dr. Tobert^2, Dr. Galina^2, Dr. Kravetski^2
Affiliations:  
^1University of Texas at Arlington, Department of Mathematics
^2United States Department of Agriculture, Agricultural Research Service in Auburn

Date: May 27, 2025

*Good Morning everyone! My name is Jose Andres Cortes, This will be my third summer with the USDA. I am currently a PhD student here at the math department, funded as a GRA by the USDA.*

# Meet the team


images: A = figs/team_photo.jpg, B = figs/ auburn_visit.jpg, C = figs/mins_machine_galina.jpg  
imagelayout:
```
        AC
        AB
```

*I am working with Dr Korzeniowski Dr. Tobert, Dr. Galina and Dr Kravetski on the MINS project. We are using gamma spectroscopy to measure soil carbon content in the field. This is the machine we're working on, this is me next to the machine when I went to visit the lab in Auburn, Alabama. The machine is called MINS, which stands for Mobile In Situ Spectroscopy.*

# Development

- USDA develops and tests the physical MINS - machine
- My role: Mathematical and statistical support
- Two main focus areas: Simulation & Data Analysis

*The physical machine and is being developed and tested by the USDA, I am responsible for providing mathematical and statistical support for the project. My work has mainly focused on two areas:*

# Simulation:

images: A: figs/mins_mcnp_slice.jpg B: figs/mins_3d.jpg

image layout:
```
AB
```
- Simulate MINS using Monte Carlo particle methods (MCNP)
- Predict machine performance in various scenarios


*First, simulation of the MINS machine using Monte Carlo paritcle methods. We use the national code MCNP. With this we simulate the MINS results, and predict the performance of the machine in different scenarios.*

# Recent work in Simulation:

image: figs/1x1x1vs7x7x7.jpg

*My most recent work on this area was to introduce functionalized soil samples. Before we would simulate the tested sample as chemically even, but ive written code that allows us to simulate the tested sample as a digitized function of spatial dimensions. This expands the data to more realistic samples.*

# Analysis:

image:figs/analysis_methods.jpg

*The second area of my work is the analysis of the data obtained from the MINS. My work last year was focused on the analysis methods for the spectrum that the machine records.*

# Goals:


1. Develop and evaluate mathematical methods for analyzing with new machine architecture.
2. Detection Range (Depth) study of the MINS machine.
3. Comparison of MINS and soil core measurements.
4. Mapping the results of the machine onto a field.
5. Estimation of the impact of the surface area sampled on field measurements.

*This summer, My scientists have asked me to focus on the following goals: I wont go through them, but overall they are about mathematically testing the capability of the machine:*
# Steps:


1. Generate Pure Spectrums (Spectrum Gneration)
2. Generate Effective Map (Associative Map)
3. Try Fast Spectrum Convolution (Spectrum Generation)
4. Compare Analysis Methods (Apply previous code to new data)
5. Variance Study
6. Depth Study
7. Core Harvesting Comparison (local)
8. Mapping Comparison
9. Field Coverage Study

*I have broken down my work into the following steps:*

# First Steps:

image: figs/detector_range.jpg

*I have already completed the first and seccond step, which was data augmentation, including a map of how the sample is scanned. This achives the goal of measuring range in simulation.*


# End Slide:

no content

*Any questions, comments, concerns? Cool, thank you*