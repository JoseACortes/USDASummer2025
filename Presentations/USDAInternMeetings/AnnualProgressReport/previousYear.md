This report outlines the first year of the project, during which we learned about physics and agriculture, designed and ran Monte-Carlo simulations for neutrons travel/scattering using the National Lab MCNP package. Granted, we developed software which implements novel soil analysis techniques, including experimental changes that we have designed.

This work was done with graduate research assistant, Jose Andres Cortes, as well as in collaboration with Galina Yakubova, Aleksandr Kavetskiy, Allen Tobert, and Sidharth Gautam. During his work, Andres showed talent, creativity, and initiative. He grasps the concepts and technologies presented to him with ease, asks relevant questions and follows up with ingenious solutions.

Work began in the summer of 2023. This period was devoted to familiarizing with the scope of the underlying Monte-Carlo simulation project along with resources acquisition. It included Andres gaining access to the Atlas Cluster, MCNP license, and the neutron travel simulator.

In the Fall, we explored time reduction methods with forced directional distribution and particle killing methods. Current simulations have run-times that span days. Although the methods explored significantly reduced time, the effects on the results varied too much from their original counterparts. We conducted simulations with varied carbon and silicone results, both locally and on the cluster.

In the spring we applied the preliminary steps of the soil analysis method, by experimenting with the regression step. Weight limiting and alternate baseline functions have made the peak finding step vastly more reliable. We also started tracking particles, which increases computation time and data output in exchange for gathering all particle data in the simulation, which helped to determine statistical effects that were not present as options in the MCNP software.

Previous implementations of the soil analysis method were performed by IGOR, a licensed software made for wave analysis.

This summer will culminate in completion of the soil analysis by developing python package and companion webapp, including alternate baseline options, with planned experimentation on the calibration model. In the following months, this software will be hosted from the UTA math office as a webapp which will be accessed by the ARS scientists. Code and resources will also be shared with ARS scientists.

At the present stage the following two new software packages (technologies) were created from scratch: (n) Calibration Tool and (nn ) Single Slice INS Denoiser.

Both packages are ready for testing by the ARS scientists. Down the line commercial release is being considered.

Although no research has yet been published, we plan to work during the summer on a joined paper with the Auburn AL based National Soil Dynamics Laboratory research scientists, and subsequently have our results published.