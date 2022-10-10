import tfs
import numpy as np
import pandas as pd

from constants import CLEANED_DPP_FILE


# Compute the plateaus via the F_RF
# If the current F_RF is the same as the last one, we're still on the plateau.
# For the tune, we need to compute the average
def append(df, df2, new_headers=None):
    res_df = pd.concat([df, df2], ignore_index=True)

    res_df = tfs.TfsDataFrame(res_df)
    res_df.headers = new_headers
    return res_df


def tune_window(data, plane):
    qx = 0.27
    qy = 0.31
    tolerance = 0.03

    if plane == 'X':
        clean = data.loc[(data < (qx + tolerance)) & (data > (qx - tolerance))]
    if plane == 'Y':
        clean = data.loc[(data < (qy + tolerance)) & (data > (qy - tolerance))]

    return clean


def remove_bad_tune_line(data, plane, low, high):
    mask = (data > low) & (data < high)
    clean = data.loc[~mask]

    removed = np.count_nonzero(mask)
    if removed:
        print(f'Removed {removed} data points from a bad line')
    return clean


def remove_bad_time(data, t0, t1):
    data['TIME'] = pd.to_datetime(data['TIME'])
    data = data.loc[~((data['TIME'] > t0) & (data['TIME'] < t1))]
    print(f'Remove data from {t0} to {t1}')
    return data


def reject_outliers(data, plane):
    data = pd.Series(data)
    data = tune_window(data, plane)
    data = remove_bad_tune_line(data, plane, 0.26, 0.268)

    Q1 = data.quantile(0.2)
    Q3 = data.quantile(0.80)
    IQR = Q3 - Q1

    fence_low = Q1 - 1.5 * IQR
    fence_high = Q3 + 1.5 * IQR
    data_cleaned = data.loc[(data > fence_low) & (data < fence_high)]

    std = np.std(data_cleaned, axis=0)
    return data_cleaned, std


def get_cleaned_tune(tunes, plane):
    cleaned_tunes, std = reject_outliers(tunes, plane)

    # if all the points are the same, the cleaned tunes would be empty
    if len(cleaned_tunes) == 0:
        if tunes.count(tunes[0]) == len(tunes):  # all the same
            return sum(tunes) / len(tunes), 0
        return None, None

    return sum(cleaned_tunes) / len(cleaned_tunes), std


def add_points(tune_x, tune_y, i, j, fp, out_tfs, data):
    # Length of plateau
    length = i - fp - 1
    # If the plateau is shorter than (arbitrary) 15 measurements, drop it
    if length < 15:
        print(f"Not logging plateau because of its short length: {i - fp - 1}")
        return out_tfs, j

    tune_avg_x, std_x = get_cleaned_tune(tune_x, 'X')
    tune_avg_y, std_y = get_cleaned_tune(tune_y, 'Y')

    if tune_avg_x is None or tune_avg_y is None:
        print(f"Not logging plateau because of equal tune data: {tune_x[0]}")
        return out_tfs, j

    # add the first point of the plateau
    d = tfs.TfsDataFrame([[data['TIME'][fp], data['F_RF'][fp], tune_avg_x, tune_avg_y, data['DPP'][fp], std_x, std_y]],
                         columns=out_tfs.columns)
    out_tfs = append(out_tfs, d, new_headers=out_tfs.headers)
    j += 1

    # And the last point
    d = tfs.TfsDataFrame(
        [[data['TIME'][i - 1], data['F_RF'][i - 1], tune_avg_x, tune_avg_y, data['DPP'][i - 1], std_x, std_y]],
        columns=out_tfs.columns)
    out_tfs = append(out_tfs, d, new_headers=out_tfs.headers)
    j += 1

    return out_tfs, j


def clean_data_for_beam(beam):
    data = tfs.read(CLEANED_DPP_FILE.format(beam=beam))
    last_frf = data['F_RF'][0]
    tune_x = []  # temporary list to hold the tune to further clean
    tune_y = []

    # Create the resulting tfs
    out_tfs = tfs.TfsDataFrame(columns=data.columns)
    headers_backup = data.headers
    out_tfs['QXERR'] = np.nan
    out_tfs['QYERR'] = np.nan

    j = 0
    fp = 0  # first point of the plateau
    # out_tfs.loc[0] = data.loc[0]

    # Clean the data given a time
    # t0 = datetime(2022, 5, 27, 20, 47, 22)
    # t1 = datetime(2022, 5, 27, 20, 49, 41)
    # data = remove_bad_time(data, t0, t1)

    data = data.reset_index(drop=True)

    for i in range(len(data.index)):
        if data['F_RF'][i] == last_frf:
            tune_x.append(data['QX'][i])
            tune_y.append(data['QY'][i])

        else:  # new plateau
            out_tfs, j = add_points(tune_x, tune_y, i, j, fp, out_tfs, data)

            # Reset the counters
            tune_x = []
            tune_y = []
            fp = i

        last_frf = data['F_RF'][i]

    # Last point
    out_tfs, _ = add_points(tune_x, tune_y, i, j, fp, out_tfs, data)

    # TFS can't write dates, convert it to str
    out_tfs = tfs.TfsDataFrame(out_tfs.astype({'TIME': str}))
    # Restore the headers
    out_tfs.headers = headers_backup

    tfs.write(output, out_tfs)
    print("\nCleaned!")