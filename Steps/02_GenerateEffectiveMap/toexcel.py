import pandas as pd
import numpy as np
import pickle
import json
import tqdm

import openpyxl

sims_file = '../01_GeneratePureSpectrums/sims_01.csv'
res_info_folder = '../01_GeneratePureSpectrums/ResInfo/res_info_'
sims_df = pd.read_csv(sims_file)
print(sims_df.head())
pickle_files = '../01_GeneratePureSpectrums/specs/'+sims_df['filename']+'.pkl'


detector_specs = {}
heat_specss = {}

for i in range(len(sims_df)):

    heat_specs = {}

    pickle_file = pickle_files[i]
    soil_resolution = sims_df['soil_resolution'][i]
    res_info_file = res_info_folder + str(soil_resolution) + '.json'
    res_info = json.load(open(res_info_file, 'r'))
    midpoints = np.array(res_info['midpoints'])
    detector_tally_ids = [int(r) for r in res_info['detector_tally_ids']]
    cell_ids = [int(r) for r in res_info['cell_ids']]
    filename = sims_df['filename'][i]
    i_bins, i_spectrums= pickle.load(open(pickle_file, 'rb'))
    energy_bins = i_bins[-1][-2]
    detector_energy_bins = i_bins[0][-2]
    detector_spec = i_spectrums[0]

    detector_specs['bins'] = detector_energy_bins
    detector_specs[filename] = detector_spec.flatten().tolist()

    causal_bins = [str(bin)[:len(str(cell_ids[0]))] for bin in i_bins[4][2]]
    causal_bins = (np.array(causal_bins)).astype(int).tolist()
    causal_spec = i_spectrums[4]
    new_shape = np.array(causal_spec.shape)
    new_shape[2] = len(cell_ids)
    new_array = np.zeros(new_shape)
    for j in range(len(cell_ids)):
        if cell_ids[j] in causal_bins:
            new_array[:, :, j] = causal_spec[:, :, causal_bins.index(cell_ids[j])]
    full_causal_spec = new_array

    full_causal_spectrums = full_causal_spec[0, 0, :, 0, 0, 0, :, 0]

    for _, cell_id in enumerate(cell_ids):
        heat_specs[cell_id] = full_causal_spectrums[_, :].flatten().tolist()

    heat_specss[filename] = heat_specs


# Create a single Excel file with multiple sheets, the first page is detector specs, the rest are heat specs per filename
with pd.ExcelWriter('heat_specs.xlsx', engine='openpyxl') as writer:
    # Write detector specs
    detector_df = pd.DataFrame(detector_specs)
    detector_df.to_excel(writer, sheet_name='Detector Specs', index=False)

    # Write heat specs for each filename
    for filename, heat_specs in tqdm.tqdm(heat_specss.items()):
        heat_df = pd.DataFrame(heat_specs)
        heat_df.to_excel(writer, sheet_name=filename, index=False)

    # # close the writer
    # writer.save()