- Describe models more clearly
- Cite soil densities and compositions
- Add more details on the simulation setup
+ Dry to 20% moisture content
- Explain mechanical mixing equation
+ Elemental component graphs
+ change weights to percentage
+ change probabilities to counts per neutron
- Change detector type
+ mention the regression model
- make points simulation and lines fitting
- explain each table

Sim GEOM

- add dimensions
- do 4 perspectives

Sim Soil

+ explain how to calculate element content
+ change density to citable
+ add moisture content
+ citable soil C content
- now cite them

PF check
- add true vs peak area graph
- add soil types to true vs predicted graph
- send fitting graphs

- explain convoluted data

---

The purpose of the paper is to consider different methods (4) of gamma spectra processing and choose the optimal (for soil C content determination).

This was done on the base of the MCNP simulation.

The simulation geometry was similar to the TNM (target neutron method, or API) setup (please, could you draw the setup geometry in Figure 1 clearer and insert some dimensions; neutron source energy: … 14 MeV; detector type should be LaBr3 or which you take for simulation? The energy resolution with energy will be different for different types of detectors, it is one of the parameters of simulation)

Soils
I suppose, you took for the simulation of the soils from Table 2. How do you calculate the element’s content for simulation (please, send me the example. Maybe it will be useful to give it in attachment). (from Table 1 are you use O? with density 0.00143? H – 0.001?)
Where you take the densities for Table 2 – give references.
I believe, densities in the Table are wet or are given for the minerals. For example, I asked Google “kaolinite soil density” and got a result: “The density of kaolinite soil typically ranges from 2.6 to 2.7 g/cm³. This is the density of the mineral itself, also known as the grain density or particle density. The bulk density, which includes the pore space between particles, will be lower, often around 1.2 to 1.5 g/cm³ for kaolin slurries.”
Note, soil C content always is given for the dry soil.
How do you calculate the density of the soil for different C content? Or do you take constant density? Note, C peak area depends on soil density. In the range about 1.1 – 1.8 g/cm3 the dependency can the assumed negligible (from the simulation results, I didn’t do simulation with dry soil density >2.0-2.4 g/cm3).
Table 3. Soil C content Natural 0-6 wt% - give reference.
Training data (put digits) – do you use it for what (for which method)?
p. 2.5. I don’t understand it. Please, could you clarify? Sentence “Linear Convolution…” and what means:” The error metric for this convolution method is based on the difference between the simulated spectral readings and the readings obtained from MCNP simulations.”

Figure 4 – is it results of which convolution? What mean “convoluted data”? How are they obtained?


3. Analysis Methods of Spectral Reading (is it mean spectra processing?)
The purpose of all methods is comparing Predicted C level with True, correct? (Figure 6, 9, 11)
Let’s consider method 1 – peak fitting.
First, to estimate the predicted C level, you have to build the calibration line.
For calculation C content both, C and Si peak areas are used.
Which soil C level do you use for calibration? Which for estimation of the error (MSE)? Did you estimate the error as square root of the square differences of the values of the blue (or green) points from Ideal fit line? Is it 7.66e-5 what unit, Cwt%?
Because you have different slope (Figure 6) for different soils, it can be assumed that calibration line for different soils is different. Did you build it? Or use one?
Green crosses I don’t understand at all, how peak area can be the same for different C level? Please, could you check and send me the raw simulated data?
Figure 5 – please, clarify the legend.


3.2. Component fitting
Here, I believe, you use soil and its component simulated spectra to define the C content.
Please, give details of which components you use for spectral signature, in which range you minimize the difference (difference or square difference?). Again, in this case we can’t consider only C peak, I believe.
Why do simulation results go above the diagonal? Need to check correctness of the calculation of the pure C peak or explain why.
Figure 9 – blue points are the same as in Figure 6? What is difference? They all are upper dash line, in Figure 6 – many below.