import pandas as pd
import numpy as np
from math import factorial
from functools import partial
from scipy.optimize import curve_fit
import tfs


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
    dpp = '\\delta'
    chroma = f'Q({dpp}) = Q_0 '
    chroma += f'+ Q\' \cdot {dpp} '

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
        p0 = np.array([0.3, 2, *[pow(10, int(o)*3) for o in range(1, order)]], dtype='float64')
        
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
