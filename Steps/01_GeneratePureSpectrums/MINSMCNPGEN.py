# takes an mcnp script ONLY the mins architecture defined as well as relevant settings and tallies
# the output is a new script with the air, and soil defined, along side relevant tallies
# take function, soil parameters 

import numpy as np


default_file = "to_copy.mcnp"

soil_limits =  -1.0, 1.0, 1.0, 1.0, -1.0, 1.0
soil_density = 1.0
soil_material = 1
soil_resolution = (10, 10, 10)

default_air_bound = 800
default_air_density = 0.0012
default_air_material = 2


elem_names = ['carbon', 'nitrogen', 'silicone']
elem_labels = ['6000', '7014', '14000']