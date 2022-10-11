from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter
from pathlib import Path
import tfs
import numpy as np
import pandas as pd
from functools import partial
import matplotlib as mpl


def plot_dpp(fig, ax, filename):
    data = tfs.read(filename)
    time = data['TIME']
    frequencies = data['F_RF']
    dpp = data['DPP']
    beam = data.headers['BEAM']

    # Convert the str time to datetime
    time = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f') for t in time]

    ax.plot(time, frequencies, label='RF Frequency')
    ax.set_title(f'DPP Change due to Frequency Change for Beam {beam}')
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
    ax.set_title(title)
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
        f = ax2.plot
    f(time, tune_x, color='red', label='$Q_x$', alpha=alpha[0])
    f(time, tune_y, color='orange', label='$Q_y$', alpha=alpha[1])
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

    # Fix the legend
    handles, labels = [], []
    for i in range(len(zp)):
        handles += zp[i][0]
        labels += zp[i][1]
    # ax.get_legend_handles_labels(),
    #                                                 ax2.get_legend_handles_labels(),
    #                                                 ax3.get_legend_handles_labels())]
    leg = ax.legend(handles, labels, loc='upper left')
    for lh in leg.legendHandles:
        lh.set_alpha(1)

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

    #fig.tight_layout()

    return xticks
