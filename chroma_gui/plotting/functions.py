from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter
import tfs
import numpy as np
import pandas as pd
from functools import partial
from sklearn.metrics import r2_score

from chroma_gui.chromaticity.chroma_fct import chromaticity_func
from chroma_gui.timber.extract import read_variables_from_csv

TUNE_Y_COLOR = "red"
TUNE_X_COLOR = "orange"

# Colors defined by the palette "deep" of Seaborn, and shuffled a bit
# This avoids using the package since only those colors are used from it
COLORS = [(0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
          (0.3333333333333333, 0.6588235294117647, 0.40784313725490196),
          (0.8666666666666667, 0.5176470588235295, 0.3215686274509804),
          (0.7686274509803922, 0.3058823529411765, 0.3215686274509804),
          (0.5058823529411764, 0.4470588235294118, 0.7019607843137254),
          (0.5764705882352941, 0.47058823529411764, 0.3764705882352941),
          (0.8549019607843137, 0.5450980392156862, 0.7647058823529411),
          (0.5490196078431373, 0.5490196078431373, 0.5490196078431373)]


def plot_dpp(fig, ax, filename):
    data = tfs.read(filename)
    time = data['TIME']
    frequencies = data['F_RF']
    dpp = data['DPP']
    beam = data.headers['BEAM']

    # Convert the str time to datetime
    time = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f') for t in time]

    ax.plot(time, frequencies, label='RF Frequency')
    #ax.set_title(f'DPP Change due to Frequency Change for Beam {beam}')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [Hz]')

    # Format the dates on the X axis

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H-%M-%S.%f'))
    # rotate and align the tick labels so they look better
    fig.autofmt_xdate()

    # Plot the tunes
    ax2 = ax.twinx()
    ax2.scatter(time, dpp, color='red', label=r'$\frac{\Delta p}{p}$')
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.6f'))
    ax2.set_ylabel(r'$\frac{\Delta p}{p}$')
    ax2.yaxis.set_ticks(np.arange(-.003, 0.002, 0.0005))

    # Fix the legend
    ax.legend(loc=2)
    ax2.legend(loc=0)


def plot_freq(fig,
              ax,
              filename,
              title,
              plot_style='scatter',
              xticks=None,
              ylim=None,
              delta_rf_flag=True,
              dpp_flag=True,
              alpha=(1, 1),
              start=None,
              end=None):
    # Plot Tune, DPP, RF and Time

    data = tfs.read(filename)

    # Restrict if given a range of time
    if start:
        data = data[pd.to_datetime(data['TIME'], format='%Y-%m-%d %H:%M:%S.%f') >= start]
    if end:
        data = data[pd.to_datetime(data['TIME'], format='%Y-%m-%d %H:%M:%S.%f') <= end]

    time = data['TIME']
    frequencies = data['F_RF']
    tune_x = data['QX']
    tune_y = data['QY']
    beam = data.headers['BEAM']
    dpp = data['DPP']

    rf0 = data.headers['F_RF']  # Nominal RF Frequency

    # Convert the str time to datetime
    time = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f') for t in time]

    zp = []  # to store the labels for legend
    # Plot the RF
    ax.plot(time, frequencies, label='RF Frequency')
    #ax.set_title(title)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [Hz]')
    zp.append(ax.get_legend_handles_labels())

    # rotate and align the tick labels so they look better
    fig.autofmt_xdate()

    # Plot the tunes
    ax2 = ax.twinx()
    if plot_style == 'scatter':
        f = partial(ax2.scatter, marker='.')
    elif plot_style == 'line':
        err_x = data['QXERR']
        err_y = data['QYERR']
        f = ax2.plot
        ax2.fill_between(time, tune_x - err_x, tune_x + err_x, color=TUNE_X_COLOR, alpha=alpha[0]/4)
        ax2.fill_between(time, tune_y - err_y, tune_y + err_y, color=TUNE_Y_COLOR, alpha=alpha[1]/4)
    f(time, tune_x, color=TUNE_X_COLOR, label='$Q_x$', alpha=alpha[0])
    f(time, tune_y, color=TUNE_Y_COLOR, label='$Q_y$', alpha=alpha[1])
    ax2.set_ylabel('Tune [$2 \pi$]')
    zp.append(ax2.get_legend_handles_labels())

    ##
    if ylim:
        ax2.set_ylim(ylim)
    ##

    if dpp_flag:
        # Add the DPP
        ax3 = ax.twinx()
        ax3.spines.right.set_position(("axes", 1.1))
        ax3.set_ylabel(r'$\frac{\Delta p}{p}$')
        ax3.plot(time, dpp, color='green', label='$\Delta p/p$', linestyle='--')
        zp.append(ax3.get_legend_handles_labels())

    # Add a ΔRF
    if delta_rf_flag:
        ax4 = ax.twinx()
        ax4.spines.left.set_position(("axes", -0.1))
        ax4.yaxis.set_ticks_position('left')
        ax4.yaxis.set_label_position('left')
        ax4.set_ylabel('$\Delta$RF [Hz]')
        delta_rf = frequencies - rf0
        ax4.plot(time, delta_rf, alpha=0)  # transparent, we only  want the axis


    # Set a higher tick frequency for the time
    xticks_freq = 10
    if not xticks:
        xticks = [t for i, t in enumerate(time) if (i % (len(time) // xticks_freq) == 0 or i == len(time) - 1)]
        if (xticks[-1] - xticks[-2]).seconds < 10:
            del xticks[-1]
        # xticks = [t for i, t in enumerate(time) if (i % (len(time) // xticks_freq) == 0)]
    ax.set_xticks(xticks)

    # Same for ΔRF, one tick per 10 Hz
    if delta_rf_flag:
        min_ = int(min(delta_rf)) // 10 * 10
        max_ = int(max(delta_rf)) // 10 * 10 + 20
        l_ = [-e for e in list(range(0, -min_, 100))] + list(range(0, max_, 100))
        ax4.set_yticks(l_)

    ax.tick_params(axis='y')
    ax2.tick_params(axis='y')
    if dpp_flag:
        ax3.tick_params(axis='y')
    if delta_rf_flag:
        ax4.tick_params(axis='y')
    ax.tick_params(axis='x')

    # Format the dates on the X axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

    # Fix the legend
    handles, labels = [], []
    for i in range(len(zp)):
        handles += zp[i][0]
        labels += zp[i][1]
    # ax.get_legend_handles_labels(),
    #                                                 ax2.get_legend_handles_labels(),
    #                                                 ax3.get_legend_handles_labels())]
    leg = ax.legend(handles, labels, loc='upper left')
    leg = ax2.legend(handles, labels, loc='upper left')
    for lh in leg.legendHandles:
        lh.set_alpha(1)

    return xticks


def plot_chromaticity(fig, ax, dpp_filename, chroma_tfs, axis, fit_orders, beam):
    """
    Plots the given orders of the chromaticity function with the values in the `chroma_tfs` TfsDataFrame
    """
    data = tfs.read(dpp_filename)

    tune = data[f'Q{axis}']
    std = data[f'Q{axis}ERR']
    dpp = data['DPP']

    # Get the chromaticity values to make the plot
    chroma_tfs = chroma_tfs[chroma_tfs['AXIS'] == axis]
    chroma_tfs = chroma_tfs[chroma_tfs['BEAM'] == beam]
    chroma_tfs = chroma_tfs.drop(['AXIS', 'BEAM'], axis=1)

    # Create the X axis
    dpp_x = np.linspace(data['DPP'].min(), data['DPP'].max())

    chi_square = 0  # Chi-square score for the fit
    for i, order in enumerate(fit_orders):
        label = f"$Q^{{({order})}}$ fit"

        # Get the values for the correct order
        chroma_to_order = chroma_tfs[chroma_tfs['UP_TO_ORDER'] == order].drop(['UP_TO_ORDER'], axis=1)

        columns = [c for c in chroma_to_order.columns if int(c[1]) <= order and 'ERR' not in c]
        chroma_values = chroma_to_order[columns].values[0]

        # Get the fit of the chromaticity and its reduced Chi-square score
        # Here we use "dpp", the same number of points as for the measurement
        model_data = chromaticity_func(dpp, *chroma_values)
        sq_residual = (tune - model_data) ** 2
        chi_square = np.sum(sq_residual / std**2)
        chi_square = chi_square / (len(tune) - len(chroma_values))  # Subtract the degrees of freedom

        # Plot the chromaticity function with the supplied values
        ax.plot(dpp_x, chromaticity_func(dpp_x, *chroma_values),
                label=label, color=COLORS[order-3], zorder=-32, linewidth=4.0)

    # Plot the measured tune with errorbars
    ax.errorbar(dpp,
                tune,
                yerr=std,
                label=f'Measurement',
                linestyle='None',
                color='black',
                elinewidth=2,
                capsize=3)

    #ax.set_title(f"Chromaticity for Beam {beam[1]}")
    ax.set_xlabel(r'$\frac{\Delta p}{p}$')
    ax.set_ylabel(f'$Q_{axis}$')
    ax.tick_params(axis="both")
    ax.legend(loc=2)

    # This will be the last computed R-square score, the highest order
    return chi_square

def plot_timber(fig, ax, filename, variables):
    timber_data = read_variables_from_csv(filename, variables)

    for i, variable in enumerate(timber_data.keys()):
        ax1 = ax.twinx()
        timestamps, values = list(zip(*timber_data[variable]))
        ax1.plot(timestamps, values, label=variable, color=f"C{i}")
        ax1.legend()


def save_chromaticity_plot(fig, filepath, formats):
    for f in formats:
        fig.savefig(f'{filepath}.{f}', format=f, pad_inches=0, bbox_inches='tight', transparent=True)
