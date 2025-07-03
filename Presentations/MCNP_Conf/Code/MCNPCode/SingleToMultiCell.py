# %%
import numpy as np
import json
import chempy as chem
import gen.scriptgen.soilconctomcnp as sm
import gen.scriptgen.miscgen as misc
import pandas as pd
from tqdm import tqdm

# %%
elem_names = ['Si', 'Al', 'H', 'Na', 'O', 'Fe', 'Mg', 'C']
atomic_numbers = [14, 13, 1, 11, 8, 26, 12, 6]
elem_labels = ['14028', '13027', '1001', '11023', '8016', '26000', '12024', '6000']
elem_densities = [2.33, 2.70, 0.001, 0.97, 0.00143, 7.87, 1.74, 2.33]

compound_names = ['SiO2', 'Al2O3', 'H2O', 'Na2O', 'Fe2O3', 'MgO', 'C']
compound_labels = [['14028', '8016'], ['13027', '8016'], ['1001', '8016'], ['11023', '8016'], ['26000', '8016'], ['12024', '8016'], ['6000']]
compound_densities = [2.65, 3.95, 1.00, 2.16, 5.24, 2.74, 2.33]
compound_mass_fractions = [chem.mass_fractions({elem_names[atomic_numbers.index(i)]: chem.Substance.from_formula(ical).composition[i] for i in chem.Substance.from_formula(ical).composition.keys()}) for ical in compound_names]

material_names = ['Silica', 'Kaolinite', 'Smectite', 'Montmorillonite', 'Quartz', 'Chlorite', 'Mica', 'Feldspar', 'Coconut']
material_labels_pername = [['SiO2', 'Al2O3'], ['SiO2', 'Al2O3', 'H2O'], ['SiO2', 'Al2O3', 'H2O'], ['SiO2', 'Al2O3', 'H2O'], ['SiO2'], ['SiO2', 'Al2O3', 'Fe2O3', 'H2O'], ['SiO2', 'Al2O3', 'H2O'], ['SiO2', 'Al2O3'], ['C']]
material_portions = [[0.97, 0.3], [0.465, 0.395, 0.14], [0.667, 0.283, 0.05], [0.435, 0.145, 0.01], [1.0], [0.25, 0.2, 0.194, 0.189], [0.464, 0.382, 0.102], [0.68, 1-0.68], [1.0]]
# aqua-calc.com, cameochemicals.noaa.gov geologyscience.com
# smecite: https://store.astm.org/gtj10971j.html#:~:text=Measured%20specific%20gravity%20values%20of%20these%20smectite%20clays%2C,essential%20bound%20water%2C%20range%20from%201.98%20to%202.14.
# montmorillonite: https://www.soilmanagementindia.com/soil-mineralogy-2/montmorillonite-structure-formation-and-uses-soil-minerals/13343#:~:text=Taking%20the%20density%20of%20soil%20solids%20as%202.7,the%20magnitude%20of%20total%20surface%20forces%20becomes%20more.
# quartz: https://webmineral.com/data/Quartz.shtml
# Chlorite: https://cameo.mfa.org/wiki/Chlorite
# Mica: https://cameo.mfa.org/wiki/Mica
# Feldspar: https://cameo.mfa.org/wiki/Feldspar
material_densities = [2.32, 3.95, 2.785, 2.7, 2.62, 2.6, 2.7, 2.55, 0.53]


# %%
def soil_characteristic_function(X, character=None, elem_labels=elem_labels):
    n = X.shape[0]
    out = np.zeros((n, len(elem_labels)))
    if character:
        elem_index = elem_labels.index(character)
        out[:, elem_index] = 1
    else:
        pass
    return out

def mass_frac_char_function(X, compound_mass_fraction, elem_labels=elem_names, compound=None):
    n = X.shape[0]
    out = np.zeros((n, len(elem_labels)))
    for key, value in compound_mass_fraction.items():
        elem_index = elem_labels.index(key)
        out[:, elem_index] = value
    return out

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
material_fns = {}
material_dns = {}

# %%
for _, material in enumerate(material_names):
    material_fns[material] = lambda X, material=material, _=_: material_function(X, material_labels_pername[_], material_portions[_], elem_labels=elem_names)
    # print(all_fns[material](np.array([[0, 0, 0]])))
    material_dns[material] = material_densities[_]

# %%
feldspar_mass_function = lambda X: material_fns['Feldspar'](X)
coconut_mass_function = lambda X: material_fns['Coconut'](X)

def mix_fn(X, c=0.0):
    # if c is a function, use it to get the portion
    cocnut_portion = (c*coconut_mass_function(X).T).T
    feldspar_portion = ((1 - c)*feldspar_mass_function(X).T).T
    return cocnut_portion + feldspar_portion

def mix_dn(X, c=0.0):
    coconut_portion = c * material_dns['Coconut']
    feldspar_portion = (1 - c) * material_dns['Feldspar']
    return coconut_portion + feldspar_portion


# %%
y1 = .10
y0 = .00
x1 = 0
x0 = -30

def lineardepth(X):
    print('shape:', X.shape)
    z = X[:, 2]
    y = (y1 - y0) / (x1 - x0) * (z - x0) + y0
    return y

# %%

_X = np.array([
    [0, 0, 0],
    [0, 0, -30]
    ])
# print(lineardepth(_X))
# print(coconut_mass_function(_X))
# print(feldspar_mass_function(_X))
mix_fn(_X, c=lineardepth(_X))

# %%
lineardepth(_X)

# %%
all_fns = {
    'Gradient': lambda X: mix_fn(X, c=lineardepth(X)),
    'Vs9': lambda X: mix_fn(X, c=np.ones_like(lineardepth(X))*0.08148730601529296),
    'Tilt': lambda X: mix_fn(X, c=np.ones_like(lineardepth(X))*0.08148730601529296),
}
alldensities = {
    'Gradient': lambda X: mix_dn(X, c=lineardepth(X)),
    'Vs9': lambda X: mix_dn(X, c=np.ones_like(lineardepth(X))*0.08148730601529296),
    'Tilt': lambda X: mix_dn(X, c=np.ones_like(lineardepth(X))*0.08148730601529296),

}

# %%
x_pad = 56
y_pad = 45
z_pad= 30

center = (0, 0, 0)

extent = (
    center[0]-x_pad, center[0]+x_pad,
    center[1]-y_pad, center[1]+y_pad,
    center[2]-z_pad, 0,
)

ress = {
    "1x1x1": (1, 1, 1), 
    "2x2x2": (2, 2, 2), 
    # "7x7x7": (7, 7, 7),
    "9x9x9": (9, 9, 9),
    }

# %%
def force_n_digits(x, n):
    # if x is less that 10^n, return 0000...x such that the length is n digits, else return x
    if x < 10**n:
        return f'{x:0{n}d}'
    return f'{x}'

# %%


# %%
sims_df = {}
id_header = "005"
count = 0

sim_folder = "compute/input/"
for res in ress:
    for f in all_fns:
        cells, cell_ids, walls, surfaces, mats, avg_sample, midpoints, sides, elems, densities, detector_tallies, detector_tally_ids = sm.make_mcnp(
            all_fns[f],
            extent,
            ress[res],
            elem_labels,
            density = alldensities[f],
            x_fix=0,
            y_fix=0,
            z_fix=-42,
            z_mul=-1,
            surface_header='200',
            cell_header='9',
            mat_header='40',
            detector_tally_header='8',
            detector_cell='101',
        )
        script = misc.mcnpgen(cells, walls, surfaces, mats, detector_tallies)

        id = id_header+str(force_n_digits(count, 3))

        label = f"{res}_{f}_{id}"
        filename = f"{label}.txt"

        sims_df[label] = {
            "soil_resolution": res,
            "function": f,
            'id': id,
            "filename": f"{label}",
        }

        elems_table = {}
        for _, id in enumerate(cell_ids):
            elems_table[id] = elems[_].flatten().tolist()
        elems_table = pd.DataFrame.from_dict(elems_table)
        elems_table.index = elem_labels
        elems_table.to_csv(f"ElemMaps/ELEMS_{label}.csv")  
        

        densities_table = {}
        for _, id in enumerate(cell_ids):
            densities_table[id] = densities[_]
        densities_table = pd.DataFrame.from_dict(densities_table, orient='index', columns=['Density'])
        densities_table.index.name = 'cell_id'
        densities_table.to_csv(f"Densities/DENSITIES_{label}.csv")        

        with open(sim_folder+filename, "w") as f:
            f.write(script)

        count += 1

    res_info = {
        'soil_resolution': res,
        'detector_tally_ids': detector_tally_ids,
        'midpoints': midpoints.tolist(),
        'cell_ids': cell_ids,
        'sides': [side.tolist() for side in sides],
        }
    with open(f"ResInfo/res_info_{res}.json", "w") as f:
        json.dump(res_info, f, indent=4)


        
sims_df = pd.DataFrame.from_dict(sims_df, orient='index')
sims_df.to_csv("sims_05.csv", index=False)


