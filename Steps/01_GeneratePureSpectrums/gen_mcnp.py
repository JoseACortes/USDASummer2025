# %%
import numpy as np

# %%
mcnp_code_start = """c CAS DETECTOR
c w/ Shielding in Place and Wheels
c
CAS DETECTOR w/ Shielding in Place and Wheels		
c		
c CELL CARDS		
c		
c @@@@@@@@@@ Detectors @@@@@@@@@@@@@@@@@		
c		
101 8 -5.08 -21 imp:n,p 1 $LaBr Detector #1 Active Region"		
c		
c @@@@@@@@@ Shielding @@@@@@@@@@@@@@@@@		
c		
121 2 -4.78 -31 imp:n,p 1 $PE_Pb"		
122 4 -11.29 -32 imp:n,p 1 $Pb1"		
123 4 -11.29 -33 imp:n,p 1 $Pb2"		
124 4 -11.29 -34 imp:n,p 1 $Pb3"		
125 4 -11.29 -35 imp:n,p 1 $Pb4"		
126 4 -11.29 -36 imp:n,p 1 $Pb5"		
127 4 -11.29 -37 imp:n,p 1 $Pb6"		
128 3 -1.5 -38 imp:n,p 1 $BA1"		
129 3 -1.5 -39 imp:n,p 1 $BA2"		
130 6 -2.7 -40 imp:n,p 1 $Al"		
131 7 -7.92 301 -302 303 -304 imp:n,p 1 $Fe-tube"		
132 5 -7.8 -401 imp:n,p 1 $Fe"		
133 5 -7.8 -402 imp:n,p 1 $Fe"		
c @@@@@@@@@@ Wheels @@@@@@@@@@@@@@@@@		
c		
21 13 -0.92 -41 42 imp:n,p 1 $Wheel 1"		
211 13 -0.92 -421 422 imp:n,p 1 $Wheel 1"		
212 13 -0.92 -423 424 imp:n,p 1 $Wheel 1"		
22 13 -0.92 -43 44 imp:n,p 1 $Wheel 2"		
221 13 -0.92 -441 442 imp:n,p 1 $Wheel 2"		
222 13 -0.92 -443 444 imp:n,p 1 $Wheel 2"		
23 13 -0.92 -45 46 imp:n,p 1 $Wheel 3"		
231 13 -0.92 -461 462 imp:n,p 1 $Wheel 3"		
232 13 -0.92 -463 464 imp:n,p 1 $Wheel 3"		
24 13 -0.92 -47 48 imp:n,p 1 $Wheel 4"		
241 13 -0.92 -481 482 imp:n,p 1 $Wheel 4"		
242 13 -0.92 -483 484 imp:n,p 1 $Wheel 4"		
c		
c @@@@@@@@@@ SOIL VOLUME @@@@@@@@@@@		
c		
"""
post_cells = """c
30 12 -0.00129 -200 #101 #121 #122 #123 #124 #125 #126 #127 #128 #129 #130 #131 #132 #133
        #21 #211 #212 #22 #221 #222 #23 #231 #232 #24 #241 #242"""  

post_walls = """ imp:n,p 1 $Rest of the World
31 0 200 imp:n,p 0 $Outside world

c SURFACE CARDS
c
c Sample Volume
"""

post_surfaces = """c Active Detector Region
21 rcc 56  -5.0 -1.0   0.0 20.3 0.0   4.5 $Detector 1 Base center, Hx,Hy,Hz, radius
c Shielding Surfaces
c Shielding Pb
31 rpp 19 29 -7.5 7.5 -11 9 $PbPE xmin xmax ymin ymax zmin zmax
32 rpp 9 19 4 9 -11 9 $Pb xmin xmax ymin ymax zmin zmax
33 rpp 9 19 -9 -4 -11 9 $Pb xmin xmax ymin ymax zmin zmax
34 rpp 19 29 7.5 12.5 -11 9 $Pb xmin xmax ymin ymax zmin zmax
35 rpp 19 29 -12.5 -7.5 -11 9 $Pb xmin xmax ymin ymax zmin zmax
36 rpp 29 34 -15 15 -11 9 $Pb xmin xmax ymin ymax zmin zmax
37 rpp 9 19 -4 4 4 9 $Pb xmin xmax ymin ymax zmin zmax
38 rpp -26 26 18 28 -11 9 $BA1 xmin xmax ymin ymax zmin zmax
39 rpp -26 26 -28 -18 -11 9 $BA2 xmin xmax ymin ymax zmin zmax
40 rpp -65 65 -28 28 10 10.5 $Al xmin xmax ymin ymax zmin zmax
401 rpp -27 27 29 34 -11 9 $Fe xmin xmax ymin ymax zmin zmax
402 rpp -27 27 -34 -29 -11 9 $Fe xmin xmax ymin ymax zmin zmax
301 px -20
302 px 18
303 cx 3.71
304 cx 3.81
c Wheels Surfaces
c 23/10.5-12 turf tires, 22.6 lb rubber
c diameter 29cm, tread width 25cm, and sidewall height 13.76 cm
c set thickness of all sides 1.3 cm (s.t. weight is 22.6 lb)
41 rcc -2 77 8   0.0 25 0.0 29 $wheel 1 outer tread surface
42 rcc -2 77 8   0.0 25 0.0 27.7 $wheel 1 inner tread surface
421 rcc -2 75.7 8   0.0 1.3 0.0 29 $wheel 1 outside sidewall exterior
422 rcc -2 75.7 8   0.0 1.3 0.0 15.24 $wheel 1 outside sidewall interior
423 rcc -2 102 8   0.0 1.3 0.0 29 $wheel 1 inside sidewall exterior
424 rcc -2 102 8   0.0 1.3 0.0 15.24 $wheel 1 inside sidewall interior
43 rcc 68 77 8   0.0 25 0.0 29 $wheel 2
44 rcc 68 77 8   0.0 25 0.0 27.7
441 rcc 68 75.7 8   0.0 1.3 0.0 29
442 rcc 68 75.7 8   0.0 1.3 0.0 15.24
443 rcc 68 102 8   0.0 1.3 0.0 29
444 rcc 68 102 8   0.0 1.3 0.0 15.24
45 rcc -2 -77 8   0.0 -25 0.0 29 $wheel 3
46 rcc -2 -77 8   0.0 -25 0.0 27.7
461 rcc -2 -75.7 8   0.0 -1.3 0.0 29
462 rcc -2 -75.7 8   0.0 -1.3 0.0 15.24
463 rcc -2 -102 8   0.0 -1.3 0.0 29
464 rcc -2 -102 8   0.0 -1.3 0.0 15.24
47 rcc 68 -77 8   0.0 -25 0.0 29 $wheel 4
48 rcc 68 -77 8   0.0 -25 0.0 27.7
481 rcc 68 -75.7 8   0.0 -1.3 0.0 29
482 rcc 68 -75.7 8   0.0 -1.3 0.0 15.24
483 rcc 68 -102 8   0.0 -1.3 0.0 29
484 rcc 68 -102 8   0.0 -1.3 0.0 15.24
200 so 800    $sphere of 800cm centered at origin
c

"""

data_card = """c DATA CARDS
mode n p
c dump every hr
prdmp -60 -60 -1
c 100 keV Neutron Energy Cutoff
Cut:n 1j 0.1
c analog neutron transport
phys:n 1j 14
phys:p
sdef pos=0 0 0 erg=14.0 vec= 0 0 1 dir=d1
si1  -1 .93 1
sp1   0  0.0  1.0
c ********************
c begin material cards
c ********************
c Pure Carbon
c ****************************************
"""

post_mats = """c **********************************
c PE+Pb, dens=4.78 g/cm3
m2      6000    -0.04286 $Carbon
        1001    -0.00714 $Hydrogen
        82000   -0.95000 $Lead
c **********************************
m3    1001 -0.048535 $Boric acid
      5010 -0.034981
      5011 -0.139923
      8016 -0.776561
c **********************************
m4    82000 1 $Lead
c **********************************
m5 26000 1 $Iron
c **********************************
m6     13027   -1.000 $ Aluminum
c **********************************
c  SS-304 (8.00 g/cm3), #298 in PNNL Compendium
m7   006000.50c -0.0004000
     014000.60c -0.0050000
     015031.50c -0.0002300
     016000.60c -0.0001500
     024000.50c -0.1899981
     025055.60c -0.0099999
     026000.50c -0.7017229
     028000.50c -0.0924991
c ****************************************
c       LaBr detector
c ****************************************
m8       35079    -0.2946       $ Br79
         35081    -0.3069       $ Br81
         57139    -0.3485       $ La139
         58140    -0.0500       $ Ce140
c ************ air den=1.15e-3g/cc
m12     8016 -0.23
        7014 -0.77
c ************ Rubber Natural den=0.92 g/cc
m13     1001 -0.118371  $weight fraction
        6000 -0.881629
c
c end material cards
c
rand seed=8674536546524321 $different initial random #; seed# - any less 18 digits
c
c ********** begin tallies *********************
fc78 *********Broadened Pulse Height Tally Sum of Detectors 1,2 & 3 ***********
e78 0 1e-5 932i 8.4295
F78:p (101)
FT78 GEB -0.026198 0.059551 -0.037176
fc18 *********Unbroadened Pulse Height Tally Sum of Detectors 1,2 & 3 ***********
e18 0 1e-5 932i 8.4295
F18:p (101)
"""
post_tallies = """
c ***********************************
c mplot tal 78 freq 10000
nps 1e9"""

# %%
z_mul = -1
mcnpgen = lambda cells, walls, surfaces, mats, tallies, detector_tallies: mcnp_code_start+cells+post_cells+" #("+str(walls[0])+" -"+str(walls[1])+" "+str(walls[2])+" -"+str(walls[3])+" "+str(z_mul*int(walls[4]))+" "+str(z_mul*-int(walls[5]))+")"+post_walls+surfaces+post_surfaces+data_card+mats+post_mats+tallies+'\n'+detector_tallies+post_tallies

# %%
default_file = "to_copy.mcnp"

soil_resolution = (10, 10, 10)

# %%
# Pure Samples to generate:
# Silica (Sand): 97% SiO2, 2% Al2O3, ~1% Other

# Kaolinite (Clay): 46.5% SiO2, 39.5% Al2O3, 14.0% H2O
# Smectite (Clay): 66.7% SiO2, 28.3% Al2O3, 5.0% H2O
# Montmorillonite (Clay): 43.5% SiO2, 14.5% Al2O3, 1.0% Na2O, 1.1% H2O, 40% Other

# Quartz (Silt): 100% SiO2

# Kaolinite (Silt): 46.5% SiO2, 39.5% Al2O3, 14.0% H2O
# Smectite (Silt): 66.7% SiO2, 28.3% Al2O3, 5.0% H2O
# Chlorite (Silt): 25% SiO2, 20% Al2O3, 19.4% Fe2O3, 18.9% H2O, 11.8% MgO, 4.9% Other

# Mica (Silt): 46.4% SiO2, 38.2% Al2O3, 10.2% H2O, 3.2% Fe2O3, 1.9% Other

# Feldspars (Silt): 68% SiO2, 20% Al2O3, 7% Other, 5% Na2O


# All the possible compounds in the soil
# SiO2, Al2O3, H2O, Na2O, Fe2O3, MgO, Other

# Possblie elements in the soil
# Si, Al, H, Na, O, Fe, Mg

# densities of the elements
# Si = 2.33 g/cm3
# Al = 2.70 g/cm3
# H = 0.001 g/cm3
# Na = 0.97 g/cm3
# O = 0.00143 g/cm3
# Fe = 7.87 g/cm3
# Mg = 1.74 g/cm3

# %%
import chempy as chem

# %%
elem_names = ['Si', 'Al', 'H', 'Na', 'O', 'Fe', 'Mg', 'C']
atomic_numbers = [14, 13, 1, 11, 8, 26, 12, 6]
elem_labels = ['14028', '13027', '1001', '11023', '8016', '26000', '12024', '6000']
elem_densities = [2.33, 2.70, 0.001, 0.97, 0.00143, 7.87, 1.74, 2.33]

# %%
# input: np.array size (n, 3). output: np.array size (n, len(elem_labels))
def soil_characteristic_function(X, character=None, elem_labels=elem_labels):
    n = X.shape[0]
    out = np.zeros((n, len(elem_labels)))
    if character:
        elem_index = elem_labels.index(character)
        out[:, elem_index] = 1
    else:
        pass
    return out

# %%
soil_characteristic_function(
    np.array([[0, 0, 0], [1, 1, 1]]), 
    character='Fe', 
    elem_labels=elem_names
    )

# %%
elem_chars_fns = {elem: None for elem in elem_names}

# %%
all_fns = {}

# %%
for elem in elem_chars_fns:
    all_fns[elem] = lambda X, elem=elem: soil_characteristic_function(X, character=elem, elem_labels=elem_names)

# %%
all_fns

# %%
compound_names = ['SiO2', 'Al2O3', 'H2O', 'Na2O', 'Fe2O3', 'MgO']
compound_labels = [['14028', '8016'], ['13027', '8016'], ['1001', '8016'], ['11023', '8016'], ['26000', '8016'], ['12024', '8016']]
compound_densities = [2.65, 3.95, 1.00, 2.16, 5.24, 2.74]
compound_mass_fractions = [chem.mass_fractions({elem_names[atomic_numbers.index(i)]: chem.Substance.from_formula(ical).composition[i] for i in chem.Substance.from_formula(ical).composition.keys()}) for ical in compound_names]

# %%
compound_mass_fractions

# %%
def mass_frac_char_function(X, compound_mass_fraction, elem_labels=elem_names):
    n = X.shape[0]
    out = np.zeros((n, len(elem_labels)))
    for key, value in compound_mass_fraction.items():
        elem_index = elem_labels.index(key)
        out[:, elem_index] = value
    return out

# %%
# print(mass_frac_char_function(np.array([[0, 0, 0], [1, 1, 1]]), compound_mass_fractions[0], elem_labels=elem_names))

# %%

for compound_i, compound in enumerate(compound_names):
    
    all_fns[compound] = lambda X, compound=compound: mass_frac_char_function(X, compound_mass_fractions[compound_i], elem_labels=elem_names)
    

# %%
all_fns

# %%
all_fns['Al2O3'](np.array([[0, 0, 0], [1, 1, 1]]))

# %%
material_names = ['Silica', 'Kaolinite', 'Smectite', 'Montmorillonite', 'Quartz', 'Chlorite', 'Mica', 'Feldspar']
material_labels_pername = [['SiO2', 'Al2O3'], ['SiO2', 'Al2O3', 'H2O'], ['SiO2', 'Al2O3', 'H2O'], ['SiO2', 'Al2O3', 'H2O'], ['SiO2'], ['SiO2', 'Al2O3', 'Fe2O3', 'H2O'], ['SiO2', 'Al2O3', 'H2O'], ['SiO2', 'Al2O3']]
material_portions = [[0.97, 0.3], [0.465, 0.395, 0.14], [0.667, 0.283, 0.05], [0.435, 0.145, 0.01], [1.0], [0.25, 0.2, 0.194, 0.189], [0.464, 0.382, 0.102], [0.68, 1-0.68]]

# %%
def material_function(
        X, 
        material_labels, 
        material_portions, 
        elem_labels=elem_names, 
        compound_labels=compound_names, 
        compound_mass_fractions=compound_mass_fractions
        ):
    n = X.shape[0]
    out = np.zeros((n, len(elem_labels)))
    for j in range(len(material_labels)):
        i = compound_labels.index(material_labels[j])


        _ = mass_frac_char_function(X, compound_mass_fractions[i], elem_labels=elem_labels)

        out += _ * material_portions[j]
    return out

# %%


# %%
material_function(np.array([[0, 0, 0], [1, 1, 1]]), material_labels_pername[0], material_portions[0], elem_labels=elem_names)

# %%

for _, material in enumerate(material_names):
    all_fns[material] = lambda X: material_function(X, material_labels_pername[_], material_portions[_], elem_labels=elem_names)
all_fns


# %%
all_fns['Silica'](np.array([[0, 0, 0], [1, 1, 1]]))

# %%
import soilconctomcnp as sm

# %%
x_pad = 56
y_pad = 45
z_pad = 92-42

center = (0, 0, 0)

extent = (
    center[0]-x_pad, center[0]+x_pad,
    center[1]-y_pad, center[1]+y_pad,
    center[2]-z_pad, 0,
)


# %%


# %%
all_fns

# %%
from tqdm import tqdm

# %%
ress = {
    "1x1x1": (1, 1, 1), 
    "2x2x2": (2, 2, 2), 
    "10x10x10": (10, 10, 10)}


# %%
import pandas as pd

# %%
def force_n_digits(x, n):
    # if x is less that 10^n, return 0000...x such that the length is n digits, else return x
    if x < 10**n:
        return f'{x:0{n}d}'
    return f'{x}'

# %%
sims_df = {}
id_header = "001"
count = 0

sim_folder = "Sims/"
for res in ress:
    for f in all_fns:
        cells, walls, surfaces, mats, avg_sample, elems, tallies, detector_tallies = sm.make_mcnp(
            all_fns[f],
            extent,
            ress[res],
            elem_labels,
            density = -2.156,
            density_map=elem_densities,
            x_fix=0,
            y_fix=0,
            z_fix=-42,
            z_mul=-1,
            surface_header='200',
            cell_header='300',
            mat_header='400',
            tally_header='500',
            detector_tally_header='808',
            detector_cell='101',
        )
        script = mcnpgen(cells, walls, surfaces, mats, tallies, detector_tallies)

        id = id_header+str(force_n_digits(count, 3))

        label = f"{res}_{f}"
        filename = f"{label}_{id}.txt"

        sims_df[label] = {
            "soil_resolution": res,
            "function": f,
            'id': id,
            "filename": filename,
        }

        with open(sim_folder+filename, "w") as f:
            f.write(script)

        count += 1
        
sims_df = pd.DataFrame.from_dict(sims_df, orient='index')
sims_df.to_csv("sims_01.csv", index=False)

# print(detector_tallies)
# print(mcnpgen(cells, walls, surfaces, mats, tallies, detector_tallies))




