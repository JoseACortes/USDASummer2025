import numpy as np
import json
import chempy as chem
from chempy.util import periodic as per
import gen.scriptgen.soilconctomcnp as sm
import gen.scriptgen.miscgen as misc
import pandas as pd
from tqdm import tqdm
from chempy import Substance

periodic_table = {per.atomic_number(sym): sym for sym in per.symbols}

material_names = ['Carbon', 'Quartz', 'Feldspar', 'Mica']
material_formulas = ['C', 'SiO2', 'NaAlO2(SiO2)3', 'K2Al2O5(Si2O5)3Al4(OH)4']
substances = [Substance.from_formula(formula) for formula in material_formulas]
compositions = [substance.composition for substance in substances]
compositions = [
	{periodic_table[atomic_number]: value for atomic_number, value in composition.items()}
	for composition in compositions
]
mass_fractions = [chem.mass_fractions(composition) for composition in compositions]

material_frame = pd.DataFrame(mass_fractions, index=material_names)