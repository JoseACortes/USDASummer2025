import os
from mcnptools import Mctal, MctalTally
import INS_Analysis as insd
import numpy as np
import pandas as pd
from tqdm import tqdm

filenames = pd.read_csv('../filenames.csv')
filenames = filenames['file_id']
filenames = [f'../compute/output/mctal/{filename}.mctal' for filename in filenames]
out_data_folder = '../data/'

# spectrograms = []
# bins = None

# for filename in tqdm(filenames):
#     bins, vals = insd.read(filename, tally=88, start_time_bin=0, end_time_bin=150, nps=1e7)
#     spectrograms.append(np.array(vals))

# spectrograms = np.array(spectrograms)
# bins = np.array(bins)

# np.savez(out_data_folder+'spectrograms.npz', x=bins, y=spectrograms)

spectrums = []
for filename in tqdm(filenames):
    bins, vals = insd.read(filename, tally=78, start_time_bin=0, end_time_bin=3, nps=1)
    
    spectrums.append(np.array(vals))

spectrums = np.array(spectrums)

np.savez(out_data_folder+'spectrums.npz', x=bins, y=spectrums)

gebless_spectrums = []
for filename in tqdm(filenames):
    bins, vals = insd.read(filename, tally=18, start_time_bin=0, end_time_bin=3, nps=1)
    
    gebless_spectrums.append(np.array(vals))

gebless_spectrums = np.array(gebless_spectrums)
bins = np.array(bins)

np.savez(out_data_folder+'gebless_spectrums.npz', x=bins, y=gebless_spectrums)
