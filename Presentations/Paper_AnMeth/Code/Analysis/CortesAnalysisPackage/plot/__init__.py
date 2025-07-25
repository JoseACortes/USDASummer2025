import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def fits_plotter(
        df, 
        c_df, 
        si_df, 
        c_bins, 
        si_bins, 
        suptitle=None, 
        output_folder=None, 
        filetype='jpg', 
        filename=None
        ):
    """
    Plot the fitting results for Carbon and Silicone spectra.
    Parameters:

        df (pd.DataFrame): DataFrame containing the fitting results.
        c_df (pd.DataFrame): DataFrame containing the Carbon fitting results.
        si_df (pd.DataFrame): DataFrame containing the Silicone fitting results.
        c_bins (np.ndarray): Array of bin edges for Carbon spectrum.
        si_bins (np.ndarray): Array of bin edges for Silicone spectrum.
        suptitle (str): Title for the entire figure.
        output_folder (str): Path to the folder where the output file will be saved.
        filetype (str): File type for the output image. Default is 'jpg'.
        filename (str): Name of the output file. If None, uses suptitle.
    Returns:

        None
    Notes:
        - The function creates a figure with two subplots for Carbon and Silicone spectra.
        - Each subplot contains three lines: True, Peak, and Baseline.
        - The resulting figure is saved as an image file with the specified filename.
    Example:
        fits_plotter(
            df=dataframe,
            c_df=carbon_dataframe,
            si_df=silicone_dataframe,
            c_bins=np.array([...]),
            si_bins=np.array([...]),
            suptitle="Fitting Results",
            output_folder="/path/to/output/"
        )
    """
    if output_folder is None:
        output_folder = './'
    if filename is None:
        filename = suptitle+'.'+filetype
    else:
        if '.' in filename:
            pass
        else:
            filename = filename+'.'+filetype

    filename = output_folder + filename

    if suptitle is None:
        suptitle = 'Fitting Results'
    else:
        suptitle = suptitle.replace(' ', '_')
        suptitle = suptitle.replace('/', '_')
        suptitle = suptitle.replace('\\', '_')
        suptitle = suptitle.replace(':', '_')
        suptitle = suptitle.replace('*', '_')
        suptitle = suptitle.replace('?', '_')
        suptitle = suptitle.replace('"', '_')
        suptitle = suptitle.replace('<', '_')
        suptitle = suptitle.replace('>', '_')
        suptitle = suptitle.replace('|', '_')
        suptitle = suptitle.replace('\'', '_')
        suptitle = suptitle.replace('\"', '_')
        suptitle = suptitle.replace(' ', '_')
        suptitle = suptitle.replace('(', '_')
        suptitle = suptitle.replace(')', '_')
        suptitle = suptitle.replace('[', '_')
        suptitle = suptitle.replace(']', '_')
        suptitle = suptitle.replace('{', '_')
        suptitle = suptitle.replace('}', '_')

        fig, axs = plt.subplots(len(df.columns), 2, figsize=(15, 10*len(df.columns)))
        fig.suptitle(suptitle, fontsize=16, fontweight='bold', y=.98)
        
        for i, col in enumerate(df.columns):
            true_spec = c_df[col+" true"]
            peak_spec = c_df[col+" peak"]
            baseline_spec = c_df[col+" baseline"]
            axs[i, 0].plot(c_bins, true_spec, label='True', marker='o')
            axs[i, 0].plot(c_bins, peak_spec, label='Peak', marker='o')
            axs[i, 0].plot(c_bins, baseline_spec, label='Baseline', marker='o')
            axs[i, 0].legend()
            axs[i, 0].set_title('Carbon Fits')
            axs[i, 0].set_xlabel('MeV')
            axs[i, 0].set_ylabel('Intensity')
            axs[i, 0].set_title(f'Carbon Fit {col}')
            
            true_spec = si_df[col+" true"]
            peak_spec = si_df[col+" peak"]
            baseline_spec = si_df[col+" baseline"]
            axs[i, 1].plot(si_bins, true_spec, label='True', marker='o')
            axs[i, 1].plot(si_bins, peak_spec, label='Peak', marker='o')
            axs[i, 1].plot(si_bins, baseline_spec, label='Baseline', marker='o')
            axs[i, 1].legend()
            axs[i, 1].set_title('Silicone Fits')
            axs[i, 1].set_xlabel('MeV')
            axs[i, 1].set_ylabel('Intensity')
            axs[i, 1].set_title(f'Silicone Fit {col}')
        # plt.show()
        plt.savefig(filename)
        plt.close(fig)

def plot_fitting_results(
        fitting_df, 
        true_peak_area, 
        column = None, 
        element='Carbon', 
        suptitle=None, 
        output_folder=None, 
        filetype='jpg'
        ):
    
    fig, axs = plt.subplots(1, 1, figsize=(5.45, 5.59), frameon=False)
    fig.suptitle(suptitle, fontsize=16, fontweight='bold', y=.98)
    axs.scatter(true_peak_area, fitting_df[element+ ' Peak Area'], label='Fitted Peak Area', marker='o', color='red')
    if column is not None:
        axs.plot(true_peak_area[column], fitting_df[element+ ' Peak Area'][column], marker='x', color='red')
    axs.legend()
    axs.set_title('Fitted Peak Area vs Element Concentration')
    axs.set_xlabel('True '+element+' %')
    axs.set_ylabel('Fitted '+element+' Peak Area')
    plt.grid()
    # plt.tight_layout()

    # plt.show()
    plt.savefig(f"{output_folder+suptitle}."+filetype)
    plt.close(fig)

def multi_spectrum(df, columns=None, bins=None, colors=None, c_window=None, si_window=None, suptitle=None, output_folder=None, large_window=[.1, np.inf], filetype='png', filename=None, show=False):
    """
    Generates a multi-panel spectrum plot with zoomed-in views of specific regions 
    and saves the resulting figure as a PNG file.
    Parameters:
        df (pd.DataFrame): The input data containing spectrum information.
        columns (list, optional): List of column names to be used from the DataFrame. 
                                  Defaults to all columns in the DataFrame.
        bins (np.ndarray): Array of bin edges corresponding to the spectrum data.
        c_window (list): Range of energy values for the Carbon peak [min, max].
        si_window (list): Range of energy values for the Silicone peak [min, max].
        suptitle (str): Title for the entire figure.
        output_folder (str): Path to the folder where the output PNG file will be saved.
        large_window (list, optional): Range of energy values for the full spectrum 
                                       [min, max]. Defaults to [.1, np.inf].
    Returns:
        None
    Notes:
        - The function creates a figure with three subplots:
            - A full spectrum plot with highlighted regions for Carbon and Silicone peaks.
            - A zoomed-in view of the Silicone peak.
            - A zoomed-in view of the Carbon peak.
        - Red rectangles and labels are used to highlight the zoomed regions in the 
          full spectrum plot.
        - Dashed blue lines connect the highlighted regions in the full spectrum plot 
          to their corresponding zoomed-in subplots.
        - The y-axis of the full spectrum plot is logarithmic.
        - The resulting figure is saved as a PNG file with the name derived from the 
          `suptitle` parameter.
    Example:
        multi_spectrum(
            df=dataframe,
            columns=['col1', 'col2'],
            bins=np.array([...]),
            c_window=[1.0, 2.0],
            si_window=[0.5, 1.0],
            suptitle="Spectrum Analysis",
            output_folder="/path/to/output/"
            """
    if columns is None:
        columns = df.columns
    if output_folder is None:
        output_folder = './'
    
    si_filter = (bins >= si_window[0]) & (bins <= si_window[1])
    c_filter = (bins >= c_window[0]) & (bins <= c_window[1])
    
    si_bins = bins[si_filter]
    c_bins = bins[c_filter]
    
    si_spec = df[columns][si_filter]
    c_spec = df[columns][c_filter]

    si_min, si_max = np.min(si_spec), np.max(si_spec)
    c_min, c_max = np.min(c_spec), np.max(c_spec)

    large_window_filter = (bins >= large_window[0]) & (bins <= large_window[1])
    large_window_bins = bins[large_window_filter]
    large_window_spec = df[columns][large_window_filter]
    fig, axs = plt.subplot_mosaic(
        """
        BAA
        CAA
        """, figsize=(8.67, 5.96), dpi=300, frameon=False)
    
    fig.suptitle(suptitle, fontsize=16, fontweight='bold', y=.95)

    if colors is None:
        if len(columns) == 1:
            colors = ['black']
        else:
            colors = plt.cm.viridis(np.linspace(0, 1, len(columns)))

    axs['A'].set_title('MINS Readings')
    for i, col in enumerate(columns):
        axs['A'].plot(large_window_bins, 
                      large_window_spec[col], label=col, color=colors[i])
    # axs['A'].legend()
    axs['A'].set_yscale('log')
    axs['A'].set_xlabel('Energy (MeV)')
    axs['A'].set_ylabel('Counts')
    # draw squares around the zoomed regions
    axs['A'].add_patch(plt.Rectangle((c_window[0], c_min), c_window[1]-c_window[0], (c_max-c_min)*1.5, fill=None, edgecolor='red'))
    axs['A'].add_patch(plt.Rectangle((si_window[0], si_min), si_window[1]-si_window[0], si_max, fill=None, edgecolor='red'))
    # label the zoomed regions
    axs['A'].text((c_window[1]+c_window[0])/2, c_max*1.5, 'Carbon', horizontalalignment='center', verticalalignment='bottom', transform=axs['A'].transData, color='red')
    axs['A'].text((si_window[1]+si_window[0])/2, si_max*1.5, 'Silicone', horizontalalignment='center', verticalalignment='bottom', transform=axs['A'].transData, color='red')

    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=axs['A'], orientation='vertical')
    cbar.set_ticks(np.linspace(0, 1, len(columns)))
    cbar.set_ticklabels(columns)

    for i, col in enumerate(columns):
        axs['B'].plot(si_bins, si_spec[col], label=col, color=colors[i])
        axs['C'].plot(c_bins, c_spec[col], label=col, color=colors[i])

    axs['B'].set_xlim(si_window[0], si_window[1])
    # axs['B'].set_ylim(si_min, si_max)
    axs['B'].set_title('Silicone Peak')
    axs['B'].set_xlabel('Energy (MeV)')
    axs['B'].set_ylabel('Counts')

    axs['C'].set_xlim(c_window[0], c_window[1])
    # axs['C'].set_ylim(c_min, c_max)
    axs['C'].set_title('Carbon Peak')
    axs['C'].set_xlabel('Energy (MeV)')
    axs['C'].set_ylabel('Counts')

    start = axs['A'].transData.transform((c_window[0], c_min))
    end = axs['C'].transData.transform((c_window[1], axs['C'].get_ylim()[0]))
    # Transform figure coordinates to display space
    inv = fig.transFigure.inverted()
    start_fig = inv.transform(start)
    end_fig = inv.transform(end)
    # Add the line in figure space
    line = Line2D(
        [start_fig[0], end_fig[0]],  # x-coordinates in figure space
        [start_fig[1], end_fig[1]],  # y-coordinates in figure space
        transform=fig.transFigure,  # Use figure transformation
        color='blue', linestyle='--', linewidth=1
    )
    fig.add_artist(line)

    start = axs['A'].transData.transform((c_window[0], c_max*1.3))
    end = axs['C'].transData.transform((c_window[1], axs['C'].get_ylim()[1]))
    # Transform figure coordinates to display space
    inv = fig.transFigure.inverted()
    start_fig = inv.transform(start)
    end_fig = inv.transform(end)
    # Add the line in figure space
    line = Line2D(
        [start_fig[0], end_fig[0]],  # x-coordinates in figure space
        [start_fig[1], end_fig[1]],  # y-coordinates in figure space
        transform=fig.transFigure,  # Use figure transformation
        color='blue', linestyle='--', linewidth=1
    )
    fig.add_artist(line)
    


    if suptitle is None:
        suptitle = 'Multi Spectrum Plot'

    if filename is None:
        filename = suptitle+'.'+filetype
    else:
        if '.' in filename:
            pass
        else:
            filename = filename+'.'+filetype

    if show:
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(f"{output_folder+filename}")
        plt.close(fig)


