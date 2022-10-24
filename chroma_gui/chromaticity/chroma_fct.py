import pandas as pd
import numpy as np
from math import factorial
from functools import partial
from scipy.optimize import curve_fit
import tfs

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


def chromaticity_func(x, *args):
    '''
        Returns the taylor expansion of the chromaticity
        q0
          + q1 * x
          + q2 * x**2 * 1/2!
          + q3 * x**3 * 1/3!
          ...
    '''
    res = 0
    for order, val in enumerate(args):
        res += val * x ** (order) * (1 / factorial(order))
    return res


def get_chromaticity_formula(order):
    dpp = r'\left( \frac{\Delta p}{p} \right)'
    chroma = f'Q {dpp} = Q_0 '
    chroma += f'+ Q\' {dpp} '

    for o in range(2, order+1):
        q_str = "Q" + "'"*o
        chroma += f'+ \\frac{{1}}{{{o}!}} {q_str} \cdot {dpp}^{o} '
    return f"${chroma}$"


def construct_chroma_tfs(fit_orders):
    max_fit_order = max(fit_orders)
    q_val = [f"Q{o}" for o in range(max_fit_order + 1)]
    q_err = [f"Q{o}_ERR" for o in range(max_fit_order + 1)]
    chroma_tfs = tfs.TfsDataFrame(columns=['AXIS', 'BEAM', 'UP_TO_ORDER', *q_val, *q_err])
    return chroma_tfs


def get_chromaticity(filename, chroma_tfs, dpp_range, fit_orders, axis):
    """
    Computes the chromaticity for a given plane and DPP file
    The values are computed via a fit, for all orders between min(fit_orders) and max(fit_orders), inclusive

    The TFS given as input is then returned with an added row containing the chromaticity values
    """
    # Print the general formula
    min_fit_order = min(fit_orders)
    max_fit_order = max(fit_orders)

    data = tfs.read(filename)
    data = data.sort_values(by=['DPP'])
    data = data[(data['DPP'] > dpp_range[0]) & (data['DPP'] < dpp_range[1])]

    # Create a list of all the fit functions, we're going to fit against all orders
    fit_funcs = list()
    for order in range(min_fit_order, max_fit_order+1):
        # Initial guesses for the chroma, Q0, Q1, then 1e3, 1e6, 1e9, etc
        p0 = [0.3, 2, *[pow(10, int(o)*3) for o in range(1, order)]]

        # Create the fit function with all the parameters
        f = partial(curve_fit, chromaticity_func, data['DPP'], data[f'Q{axis}'], p0=p0)
        # Apply the errors to the fit if we got some
        if data[f'Q{axis}ERR'].all() != 0:
            f = partial(f, sigma=data[f'Q{axis}ERR'])
        fit_funcs.append(f)

    # Finally call the function and store the result!
    for i, fit_func in enumerate(fit_funcs):
        popt, pcov = fit_func()
        std = np.sqrt(np.diag(pcov))

        # Populate the chromaticity TFS
        order = i + min_fit_order
        remaining = [0] * ((max_fit_order - min_fit_order) - (len(popt)-(min_fit_order+1)))  # we have Q0, so +1

        new_data = tfs.TfsDataFrame([[axis, data.headers['BEAM'], order, *popt, *remaining, *std, *remaining]],
                                    columns=chroma_tfs.columns)
        chroma_tfs = pd.concat([chroma_tfs, new_data], ignore_index=True)

    chroma_tfs.headers['MIN_FIT_ORDER'] = min(fit_orders)
    chroma_tfs.headers['MAX_FIT_ORDER'] = max(fit_orders)

    return chroma_tfs


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

    for i, order in enumerate(fit_orders):
        label = f"$Q^{{({order})}}$ fit"

        # Get the values for the correct order
        chroma_to_order = chroma_tfs[chroma_tfs['UP_TO_ORDER'] == order].drop(['UP_TO_ORDER'], axis=1)

        columns = [c for c in chroma_to_order.columns if int(c[1]) <= order and 'ERR' not in c]
        chroma_values = chroma_to_order[columns].values[0]

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

    ax.set_title(f"Chromaticity for Beam {beam[1]}")
    ax.set_xlabel(r'$\frac{\Delta p}{p}$', fontsize=15)
    ax.set_ylabel(f'$Q_{axis}$', fontsize=15)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(loc=2)


def get_maximum_chromaticity(chroma_tfs):
    df = chroma_tfs[chroma_tfs['UP_TO_ORDER'] == chroma_tfs['UP_TO_ORDER'].max()]
    df = df.drop('UP_TO_ORDER', axis=1)
    return df


def get_chromaticity_df_with_notation(chroma_tfs):
    """
        Returns a dataFrame with the chromaticity with the headers set with exponents and the values divided
    """

    max_order = chroma_tfs.headers['MAX_FIT_ORDER']
    headers = ['BEAM', 'AXIS']
    for order in range(max_order+1):
        prime = f"({order})" if order > 0 else ""
        power = (order - 1) * 3 if order > 0 else 0
        if order == 0:
            headers.append('Q')
        elif order == 1:
            headers.append(f"Q^{prime}")
        else:
            headers.append(f"Q^{prime} [x10^{power}]")

    values = []
    for index, row in chroma_tfs.iterrows():
        beam = row['BEAM']
        axis = row['AXIS']

        new_row = [beam, axis]
        for order in range(max_order+1):
            power = (order - 1) * 3 if order > 0 else 0
            val = round(row[f'Q{order}'] / 10 ** power, 2)
            err = round(row[f'Q{order}_ERR'] / 10 ** power, 2)
            new_row.append(rf"{val} ± {err}")

        values.append(new_row)

    new_tfs = pd.DataFrame(values, columns=headers)

    return new_tfs
