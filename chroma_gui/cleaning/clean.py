import tfs
import numpy as np
import pandas as pd
import logging
from scipy import signal
import NAFFlib

from dateutil.parser import isoparse
from pathlib import Path

logger = logging.getLogger('chroma_GUI - Cleaning')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# Compute the plateaus via the F_RF
# If the current F_RF is the same as the last one, we're still on the plateau.
# For the tune, we need to compute the average
def append(df, df2, new_headers=None):
    res_df = pd.concat([df, df2], ignore_index=True)

    res_df = tfs.TfsDataFrame(res_df)
    res_df.headers = new_headers
    return res_df


def tune_window(data, plane, qx, qy):
    """
    Returns the tune data without the points outside the defined windows
    Arguments:
        - qx: tuple of lower and upper bounds for Qx, e.g. (0.26, 0.28)
        - qy: tuple of lower and upper bounds for Qy
    """
    if plane == 'X':
        clean = data.loc[(data < (qx[1])) & (data > (qx[0]))]
    if plane == 'Y':
        clean = data.loc[(data < (qy[1])) & (data > (qy[0]))]

    return clean


def remove_bad_tune_line(data, plane, low, high):
    mask = (data > low) & (data < high)
    clean = data.loc[~mask]

    removed = np.count_nonzero(mask)
    if removed:
        logger.info(f'Removed {removed} data points from a bad line')
    return clean


def remove_bad_time(data, t0, t1):
    data['TIME'] = pd.to_datetime(data['TIME'])
    data = data.loc[~((data['TIME'] > t0) & (data['TIME'] < t1))]
    logger.info(f'Removed data from {t0} to {t1}')
    return data


def reject_outliers(data, plane, qx_window, qy_window, quartiles, bad_tunes):
    data = pd.Series(data)
    data = tune_window(data, plane, qx_window, qy_window)

    for q0, q1 in bad_tunes:
        data = remove_bad_tune_line(data, plane, q0, q1)

    Q1 = data.quantile(quartiles[0])
    Q3 = data.quantile(quartiles[1])
    IQR = Q3 - Q1

    fence_low = Q1 - 1.5 * IQR
    fence_high = Q3 + 1.5 * IQR
    data_cleaned = data.loc[(data > fence_low) & (data < fence_high)]

    std = np.std(data_cleaned, axis=0)
    return data_cleaned, std


def get_cleaned_tune(tunes, plane, qx_window, qy_window, quartiles, bad_tunes):
    cleaned_tunes, std = reject_outliers(tunes, plane, qx_window, qy_window, quartiles, bad_tunes)

    # if all the points are the same, the cleaned tunes would be empty
    if len(cleaned_tunes) == 0:
        if tunes.count(tunes[0]) == len(tunes):  # all the same
            return sum(tunes) / len(tunes), 0
        return None, None

    return sum(cleaned_tunes) / len(cleaned_tunes), std


def merge_overlap(array):
    """ Merge the overlapping arrays contained in the given array """
    # Initialize and empty array of the maximum possible size
    res = np.zeros(len(array) * len(array[0]), dtype=np.float64)

    res_counter = 0  # position of the last inserted data in res
    for next_array in array:
        n_res = res_counter
        n = min(n_res, len(next_array))  # should be 2048 anyway

        j = 0  # to iterate over res
        for i in range(1, n+1):  # to iterate over next_array
            if next_array[n-i] == res[n_res - 1 - j]:
                j += 1
            else:
                j = 0

        data_to_add = next_array[j:]
        res[res_counter:res_counter+len(data_to_add)] = data_to_add

        res_counter += len(data_to_add)

    return res[:res_counter]


def get_spectrogram(raw_data, start_plateau, end_plateau, variables, seconds_step):
    """ Divide the plateau in chunks of x seconds and apply a spectrogram on it """
    # Create a mask to only get the data in the plateau
    d = raw_data.loc[variables + 'H']
    mask = d['TIMESTAMP'].astype('float') >= start_plateau.timestamp()
    mask = mask & (d['TIMESTAMP'].astype('float') <= end_plateau.timestamp())

    # Create chunks x seconds long
    chunks = int((end_plateau - start_plateau).seconds / seconds_step)
    if chunks == 0:
        t = (end_plateau - start_plateau).seconds
        logger.warning(f"The plateau is only {t} seconds long. It will be analyzed in one chunk")
        chunks = 1

    f, t, Sxx = {}, {}, {}
    merged_data = {}
    # Do the spectrogram analysis for each plane
    for plane in ('H', 'V'):
        # Merge the overlapping data returned by Timber
        merged_data[plane] = merge_overlap(raw_data.loc[variables + plane]['VALUE'][mask])
        elements_per_chunk = int(len(merged_data[plane]) / chunks)

        # print(f'  Plateau will be analyzed in {chunks} chunks, each of {seconds_step} seconds ({seconds_step*11_000} turns)')
        f[plane], t[plane], Sxx[plane] = signal.spectrogram(merged_data[plane],
                                                            nperseg=elements_per_chunk,
                                                            noverlap=elements_per_chunk // 8,
                                                            fs=1)
    return f, t, Sxx


def get_avg_tune_from_naff(raw_data, start_plateau, end_plateau, variables, seconds_step, qx_window, qy_window,
                           bad_tunes):
    """ Divide the plateau in chunks of x seconds and use NAFF on it"""
    # Create a mask to only get the data in the plateau
    d = raw_data.loc[variables + 'H']
    mask = d['TIMESTAMP'].astype('float') >= start_plateau.timestamp()
    mask = mask & (d['TIMESTAMP'].astype('float') <= end_plateau.timestamp())
    window = {'H': qx_window, 'V': qy_window}

    # Create chunks x seconds long
    chunks = int((end_plateau - start_plateau).seconds / seconds_step)
    if chunks == 0:
        t = (end_plateau - start_plateau).seconds
        logger.warning(f"The plateau is only {t} seconds long. It will be analyzed in one chunk")
        chunks = 1

    merged_data = {}
    tunes = {'H': [], 'V': []}
    # Get the tune via NAFF for each plane
    for plane in ('H', 'V'):
        # Merge the overlapping data returned by Timber
        merged_data[plane] = merge_overlap(raw_data.loc[variables + plane]['VALUE'][mask])
        elements_per_chunk = int(len(merged_data[plane]) / chunks)

        # Process each chunk
        for i in range(chunks):
            data = merged_data[plane][elements_per_chunk * i: elements_per_chunk * (i+1)]

            #spectrum = pnf.naff(data,
            #                    turns=len(data)-1,  # somehow it fails with all the data
            #                    nterms=20,
            #                    skipTurns=0,
            #                    getFullSpectrum=False,
            #                    window=1)

            spectrum, _, _ = NAFFlib.get_tunes(data, 20)

            for frequency in spectrum:
                in_window = window[plane][0] <= frequency <= window[plane][1]
                not_bad_line = not np.any([(bad_low <= frequency <= bad_high) for bad_low, bad_high in bad_tunes])
                if in_window and not_bad_line:
                    tunes[plane].append(frequency)
                    break

        # Discard the measurement for both planes if there's no usable data
        if len(tunes[plane]) == 0:
            #tunes[plane] = (None, None)
            return {'H': (None, None), 'V': (None, None)}


    tune_x = np.average(tunes['H']), np.std(tunes['H'])
    tune_y = np.average(tunes['V']), np.std(tunes['V'])
    print(tune_x, tune_y)
    return {'H': tune_x, 'V': tune_y}


def get_max_peak(x_data, y_data, plane, window, bad_tunes):
    # Plot the maximum peaks for each plane
    # Find the peaks via scipy
    peaks, _ = signal.find_peaks(y_data,  # power density data
                                 distance=100,  # minimum distance between peaks
                                 )
    # Get a window to only get the interesting peaks
    tune_window = (x_data[peaks] >= window[plane][0]) & (x_data[peaks] <= window[plane][1])

    # Remove the bad lines
    for bad_low, bad_high in bad_tunes:
        mask = ~((x_data[peaks] > bad_low) & (x_data[peaks] < bad_high))
        tune_window = tune_window & mask

    # Get the peak amplitude associated to a tune (e.g. 2.6)
    # Sort it in reverse order
    peak_amp, tunes = zip(*sorted(zip(y_data[peaks][tune_window],
                                      x_data[peaks][tune_window]),
                                  reverse=True))
    return tunes[0]


def get_avg_tune_from_spectrogram(f, Sxx, kernel_size, qx_window, qy_window, bad_tunes):
    tunes = {'H': [], 'V': []}
    avg_tunes = {'H': [], 'V': []}
    for plane in 'H', 'V':
        # print(f'Plane {plane}:')
        # Iterate on the segments
        for i in range(len(Sxx[plane][0])):
            # Filter the data
            data = signal.medfilt(Sxx[plane][:, i],
                                  kernel_size=kernel_size)
            # Get the tune
            window = {'H': qx_window,
                      'V': qy_window
                      }
            tune = get_max_peak(f[plane], data, plane, window, bad_tunes)
            tunes[plane].append(tune)

        avg_tunes[plane] = [np.mean(tunes[plane]),
                            np.std(tunes[plane])]
    return avg_tunes


def add_points(tune_x, tune_y, i, j, fp, out_tfs, data, qx_window, qy_window, quartiles, plateau_length, bad_tunes,
               method="bbq", raw_bbq_data=None, seconds_step=None, kernel_size=None, beam=None, output_path=None):
    """
        Gets the tune from one of the defined methods and adds it to the TFS
    """
    # Length of plateau
    length = i - fp - 1
    # If the plateau is shorter than (arbitrary) 15 measurements, drop it
    if length < plateau_length:
        logger.debug(f"Not logging plateau because of its short length: {i - fp - 1}")
        return out_tfs, j

    # Use the selected method to compute the tune
    if method == "bbq":  # Use the already processed tune from TIMBER
        tune_avg_x, std_x = get_cleaned_tune(tune_x, 'X', qx_window, qy_window, quartiles, bad_tunes)
        tune_avg_y, std_y = get_cleaned_tune(tune_y, 'Y', qx_window, qy_window, quartiles, bad_tunes)
    elif method == "raw_bbq_spectrogram":  # Do our own magic on the raw BBQ data using a spectrogram
        start_plateau = isoparse(data['TIME'][fp])
        end_plateau = isoparse(data['TIME'][i - 1])
        variables = f'LHC.BQBBQ.CONTINUOUS_HS.B{beam}:ACQ_DATA_'
        f, t, Sxx = get_spectrogram(raw_bbq_data, start_plateau, end_plateau, variables, seconds_step)
        tunes_from_raw = get_avg_tune_from_spectrogram(f, Sxx, kernel_size, qx_window, qy_window, bad_tunes)
        tune_avg_x, std_x = tunes_from_raw['H']
        tune_avg_y, std_y = tunes_from_raw['V']
    elif method == "raw_bbq_naff":
        start_plateau = isoparse(data['TIME'][fp])
        end_plateau = isoparse(data['TIME'][i - 1])
        variables = f'LHC.BQBBQ.CONTINUOUS_HS.B{beam}:ACQ_DATA_'
        tunes_from_raw = get_avg_tune_from_naff(raw_bbq_data, start_plateau, end_plateau, variables, seconds_step,
                                                qx_window, qy_window, bad_tunes)
        tune_avg_x, std_x = tunes_from_raw['H']
        tune_avg_y, std_y = tunes_from_raw['V']

    # Reject short plateaus that have no std
    if tune_avg_x is None or tune_avg_y is None:
        logger.debug(f"Not logging plateau because of equal tune data: {tune_x[0]}")
        logger.debug(f"  Time: {data['TIME'][fp]} / {data['TIME'][i-1]}")
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


def clean_data_for_beam(input_file, output_path, output_file, qx_window, qy_window, quartiles, plateau_length,
                        bad_tunes, method, raw_bbq_file=None, seconds_step=None, kernel_size=None, beam=None,
                        signal=None):
    data = tfs.read(input_file)
    last_frf = data['F_RF'][0]

    if method == "bbq":  # can be "bbq", "raw_bbq_naff", "raw_bbq_spectrogram"
        raw_data = None
    else:
        #raw_data = pd.read_pickle(raw_bbq_file)
        raw_data = pd.read_hdf(raw_bbq_file)

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
            out_tfs, j = add_points(tune_x, tune_y, i, j, fp, out_tfs, data, qx_window, qy_window, quartiles,
                                    plateau_length, bad_tunes, method, raw_data, seconds_step, kernel_size, beam,
                                    output_path)

            # Reset the counters
            tune_x = []
            tune_y = []
            fp = i

        last_frf = data['F_RF'][i]

        # Fire the progress signal to update the GUI, as a rounded down percentage
        signal.emit(np.floor(i / len(data.index) * 100))

    # Last point
    out_tfs, _ = add_points(tune_x, tune_y, i, j, fp, out_tfs, data, qx_window, qy_window, quartiles,
                            plateau_length, bad_tunes, method, raw_data, seconds_step, kernel_size, beam, output_path)

    # TFS can't write dates, convert it to str
    out_tfs = tfs.TfsDataFrame(out_tfs.astype({'TIME': str}))
    # Restore the headers
    out_tfs.headers = headers_backup

    if beam is not None:
        out_tfs.headers['BEAM'] = f'B{beam}'

    tfs.write(output_path / output_file, out_tfs)
