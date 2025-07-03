import pandas as pd
import numpy as np
import pickle
import json

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# files

sims_file = '../01_GeneratePureSpectrums/sims_01.csv'
res_info_folder = '../01_GeneratePureSpectrums/ResInfo/res_info_'
sims_df = pd.read_csv(sims_file)
pickle_files = '../01_GeneratePureSpectrums/specs/'+sims_df['filename']+'.pkl'


filenames = [sims_df['filename'][i] for i in range(len(sims_df))]
res_info_files = [res_info_folder + str(soil_resolution) + '.json' for soil_resolution in sims_df['soil_resolution']]
res_infos = [json.load(open(res_info_file, 'r')) for res_info_file in res_info_files]
sidess = [res_info['sides'] for res_info in res_infos]


filenames = []
n_90 = []
n_95 = []
n_99 = []
n_100 = []
n_total = []

clouds = {
    '90': [],
    '95': [],
    '99': [],
    '100': []
}

cell_portions = []

for i in range(len(sims_df)):

    pickle_file = pickle_files[i]
    soil_resolution = sims_df['soil_resolution'][i]
    res_info_file = res_info_folder + str(soil_resolution) + '.json'
    res_info = json.load(open(res_info_file, 'r'))
    midpoints = np.array(res_info['midpoints'])
    detector_tally_ids = [int(r) for r in res_info['detector_tally_ids']]
    cell_ids = [int(r) for r in res_info['cell_ids']]
    filename = sims_df['filename'][i]
    i_bins, i_spectrums= pickle.load(open(pickle_file, 'rb'))
    detector_energy_bins = i_bins[0][-2]
    detector_spec = i_spectrums[0]
    causal_bins = [str(bin)[:len(str(cell_ids[0]))] for bin in i_bins[4][2]]
    causal_bins = (np.array(causal_bins)).astype(int).tolist()
    causal_energy_bins = i_bins[4][-2]
    causal_spec = i_spectrums[4]
    new_shape = np.array(causal_spec.shape)
    new_shape[2] = len(cell_ids)
    new_array = np.zeros(new_shape)
    for j in range(len(cell_ids)):
        if cell_ids[j] in causal_bins:
            new_array[:, :, j] = causal_spec[:, :, causal_bins.index(cell_ids[j])]
    full_causal_spec = new_array
    cell_heating = full_causal_spec[0, 0, :, 0, 0, 0, :, 0]
    total_soil_heating = np.sum(cell_heating, axis=0)
    cells_over_total_heating = np.nan_to_num(np.divide(cell_heating, total_soil_heating), copy=False, nan=0.0, posinf=None, neginf=None)
    cells_over_total_heating = cells_over_total_heating/np.sum(cells_over_total_heating)
    cell_portion = np.sum(cells_over_total_heating, axis=1)

    sorted_indicies = np.argsort(cell_portion)[::-1]
    sorted_vals = (cell_portion)[sorted_indicies]
    def percent_cloud(sorted_indicies, sorted_vals, threshold=0.90):
        total = np.sum(sorted_vals)
        cumulative_sum = np.cumsum(sorted_vals)
        percent = cumulative_sum / total
        return sorted_indicies[percent < threshold]
    cell_portion = np.sum(cells_over_total_heating, axis=1)

    cell_portions.append(cell_portion)

    filenames.append(filename)
    
    cloud_90 = percent_cloud(sorted_indicies, sorted_vals, threshold=.90)
    clouds['90'].append(cloud_90)
    n_90.append(len(cloud_90))

    cloud_95 = percent_cloud(sorted_indicies, sorted_vals, threshold=.95)
    clouds['95'].append(cloud_95)
    n_95.append(len(cloud_95))

    cloud_99 = percent_cloud(sorted_indicies, sorted_vals, threshold=.99)
    clouds['99'].append(cloud_99)
    n_99.append(len(cloud_99))

    cloud_100 = percent_cloud(sorted_indicies, sorted_vals, threshold=1.0)
    clouds['100'].append(cloud_100)
    n_100.append(len(cloud_100))

    n_total.append(len(sorted_indicies))

results_df = pd.DataFrame({
    'filename': filenames,
    'n_90': n_90,
    'n_95': n_95,
    'n_99': n_99,
    'n_100': n_100,
    'n_total': n_total
})

clouds_df = pd.DataFrame({
    'filename': filenames,
    'cloud_90': clouds['90'],
    'cloud_95': clouds['95'],
    'cloud_99': clouds['99'],
})

results_df.to_csv('n_cells_over_total_heating.csv', index=False)
print(results_df)

import plotting as pltt


def cloud_from_cloud(
        ax0, cloud, filter_mask, 
        alpha=1, 
        color='green',
        zorder=1002,
        ):
    
    cell_mask_vox = pltt.vox_maker(
        cloud, 
        midpoints, 
        cell_portions[22]
        )
    
    cell_mask_vox = cell_mask_vox & filter_mask
    
    pltt.concave_hull(
        ax0, 
        cell_mask_vox, 
        sidess[22], 
        color=color, 
        alpha=alpha,
        zorder=zorder
        )
    

def cloud3(ax, filter_mask):
    cloud_from_cloud(
        ax, 
        clouds['99'][22], 
        filter_mask, 
        alpha=1, 
        color='red',
        zorder=1002,
        )


    cloud_from_cloud(
        ax, 
        clouds['95'][22], 
        filter_mask, 
        alpha=1, 
        color='orange',
        zorder=1002,
        )


    cloud_from_cloud(
        ax, 
        clouds['90'][22], 
        filter_mask, 
        alpha=1, 
        color='yellow',
        zorder=1002,
        )



soil_res = [int(s) for s in soil_resolution.split('x')]

fig, axs = plt.subplots(1, 3, subplot_kw={'projection': '3d'}, figsize=(8*3, 8))
fig.suptitle(f'Cell Portion of Total Heating for {filenames[22]}', fontsize=10)

# side
ax0 = axs[0]
ax0.set_title('Side View', fontsize=10)
pltt.plot_box(ax0, -56, 56, -45, 45, 42, 92, color='brown', alpha=.5, label='Soil', zorder=1)
ax0.set_ylim([0, 50])
filter_mask = (midpoints[:, 1]>-0.1).reshape(soil_res)
cloud3(ax0, filter_mask)

pltt.plot_cone(ax0, pos=(0, 0, 0), vec=(0.0, 0, 1.0), dir=np.degrees(np.cos(.95)), length=40, color='blue', alpha=.5, label='Emitter Cone', zorder=2000)
pltt.plot_cylinder(ax0, base=(56, -5.0, -1.0), vec=(0.0, 20.3, 0.0), radius=4.5, height=20.3, color='blue', alpha=1, label='Detector', zorder=200101)
pltt.plot_box(ax0, 29, 34, -15, 15, -11, 9, color='black', alpha=1, zorder=200100, label='Shielding')
pltt.extra(ax0)
ax0.view_init(elev=0, azim=-90,)
ax0.set_yticklabels([])
ax0.set_yticks([]) 
ax0.set_ylabel('')
ax0.set_aspect('equal')
plt.legend()

# top
ax1 = axs[1]
ax1.set_title('Top View', fontsize=10)
pltt.plot_box(ax1, -56, 56, -45, 45, 42, 92, color='brown', alpha=.5, label='Soil', zorder=1)
ax1.set_zlim([42.1, 100])
filter_mask = np.ones(soil_res, dtype=bool)
cloud3(ax1, filter_mask)
pltt.plot_cone(ax1, pos=(0, 0, 0), vec=(0.0, 0, 1.0), dir=np.degrees(np.cos(.95)), length=40, color='blue', alpha=.5, label='Emitter Cone', zorder=2000)
pltt.plot_cylinder(ax1, base=(56, -5.0, -1.0), vec=(0.0, 20.3, 0.0), radius=4.5, height=20.3, color='blue', alpha=1, label='Detector', zorder=200101)
pltt.plot_box(ax1, 29, 34, -15, 15, -11, 9, color='black', alpha=1, zorder=200100, label='Shielding')
pltt.extra(ax1)
ax1.view_init(elev=90, azim=-90,)
ax1.set_zticklabels([])
ax1.set_zticks([])
ax1.set_zlabel('')
ax1.set_aspect('equal')

# front
ax2 = axs[2]
ax2.set_title('Front View', fontsize=10)
pltt.plot_box(ax2, -56, 56, -45, 45, 42, 92, color='brown', alpha=.5, label='Soil', zorder=1)
ax2.set_xlim([-60, 15])
filter_mask = (midpoints[:, 0]<20).reshape(soil_res)
cloud3(ax2, filter_mask)
pltt.plot_cone(ax2, pos=(0, 0, 0), vec=(0.0, 0, 1.0), dir=np.degrees(np.cos(.95)), length=40, color='blue', alpha=.5, label='Emitter Cone', zorder=2000)
pltt.plot_cylinder(ax2, base=(56, -5.0, -1.0), vec=(0.0, 20.3, 0.0), radius=4.5, height=20.3, color='blue', alpha=1, label='Detector', zorder=200101)
pltt.plot_box(ax2, 29, 34, -15, 15, -11, 9, color='black', alpha=1, zorder=200100, label='Shielding')
pltt.extra(ax2)
ax2.view_init(elev=0, azim=0,)
ax2.set_xticklabels([])
ax2.set_xticks([])
ax2.set_xlabel('')
ax2.set_aspect('equal')

plt.savefig('figs/clouds/clouds_99_95_90.png', bbox_inches='tight', dpi=300)