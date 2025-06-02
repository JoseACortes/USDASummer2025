# %%
from mcnptools import Mctal, MctalTally

# %%
import pandas as pd
import numpy as np
import tqdm
import pickle
import os
import json

# %%
sims_df_file = '../sims_03.csv'
sims_df = pd.read_csv(sims_df_file)

mctal_folder = '../compute/output/mctal/'
mctal_files = mctal_folder+sims_df['filename']+'.mctal'


res_info_folder = '../ResInfo/'
specs_folder = '../specs/'

# %%
res_info_files = [f for f in os.listdir(res_info_folder) if f.endswith('.json')]
res_info_files = [os.path.join(res_info_folder, f) for f in res_info_files]
res_info_labels = [f.split('/')[-1].split('.')[0] for f in res_info_files]
res_info_labels = [f.split('_')[-1] for f in res_info_labels]

res_infos = {}
for label, f in zip(res_info_labels, res_info_files):
    with open(f, 'r') as json_file:
        res_infos[label] = json.load(json_file)


# %%
def readMCTAL(file, tally):

    m = Mctal(file)
    tfc = MctalTally.TFC
    # print(tfc)
    t = m.GetTally(tally)

    t_f_bins = t.GetFBins()
    t_d_bins = t.GetDBins()
    t_u_bins = t.GetUBins()
    t_s_bins = t.GetSBins()
    t_m_bins = t.GetMBins()
    t_c_bins = t.GetCBins()
    t_e_bins = t.GetEBins()
    t_t_bins = t.GetTBins()

    bins = [t_f_bins, t_d_bins, t_u_bins, t_s_bins, t_m_bins, t_c_bins, t_e_bins, t_t_bins]
    shape = (len(t_f_bins), len(t_d_bins), len(t_u_bins), len(t_s_bins), len(t_m_bins), len(t_c_bins), len(t_e_bins), len(t_t_bins))
    # print(shape)
    array = np.zeros(shape)
    for i in range(len(t_f_bins)):
        for j in range(len(t_d_bins)):
            for k in range(len(t_u_bins)):
                for l in range(len(t_s_bins)):
                    for m in range(len(t_m_bins)):
                        for n in range(len(t_c_bins)):
                            for o in range(len(t_e_bins)):
                                for p in range(len(t_t_bins)):
                                    array[i,j,k,l,m,n,o,p] = t.GetValue(i,j,k,l,m,n,o,p)
    # #                        f    d    u    s    m    c   e   t
    return bins, array

# %%
# for each simulation, read the mctal file and save the results in a .pkl file
for i in tqdm.tqdm(range(len(sims_df)), desc="Processing simulations"):
    sim_label = sims_df['filename'][i]
    tally_bins = []
    tally_arrays = []
    for tally in tqdm.tqdm(res_infos[sims_df['soil_resolution'][i]]['detector_tally_ids'], desc="Processing tallies", leave=False):
        # print(mctal_files[i])
        i_bins, i_array = readMCTAL(mctal_files[i], int(tally))
        tally_bins.append(i_bins)
        tally_arrays.append(i_array)
    
    # save the results in a .pkl file
    with open(specs_folder+sim_label+'.pkl', 'wb') as f:
        pickle.dump([tally_bins, tally_arrays], f)