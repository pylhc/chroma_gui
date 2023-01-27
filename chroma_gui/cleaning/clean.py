import tfs
import numpy as np
import pandas as pd
import logging
from scipy import signal
import PyNAFF as pnf
from dateutil.parser import isoparse
import sdds
from omc3 import hole_in_one
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


def save_data_to_lhc_sdds(timestamp, data_x, data_y, bpm_name, path):
    # IDs
    N_BUNCHES: str = "nbOfCapBunches"
    BUNCH_ID: str = "BunchId"
    HOR_BUNCH_ID: str = "horBunchId"
    N_TURNS: str = "nbOfCapTurns"
    ACQ_STAMP: str = "acqStamp"
    BPM_NAMES: str = "bpmNames"

    POSITIONS = {
        "X": "horPositionsConcentratedAndSorted",
        "Y": "verPositionsConcentratedAndSorted",
    }

    definitions = [
        sdds.classes.Parameter(ACQ_STAMP, "llong"),
        sdds.classes.Parameter(N_BUNCHES, "long"),
        sdds.classes.Parameter(N_TURNS, "long"),
        sdds.classes.Array(BUNCH_ID, "long"),
        sdds.classes.Array(BPM_NAMES, "string"),
        sdds.classes.Array(POSITIONS["X"], "float"),
        sdds.classes.Array(POSITIONS["Y"], "float"),
    ]
    values = [
        timestamp,
        1,
        len(data_x),
        [0] * len(data_x),
        [bpm_name],
        np.ravel(data_x),
        np.ravel(data_y),
    ]
    sdds.write(sdds.SddsFile("SDDS1", None, definitions, values), path)


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


def get_avg_tune_from_naff(raw_data, start_plateau, end_plateau, variables, seconds_step, qx_window, qy_window):
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

            spectrum = pnf.naff(data,
                                turns=len(data)-1,  # somehow it fails with all the data
                                nterms=2,
                                skipTurns=0,
                                getFullSpectrum=False,
                                window=1)

            tune = None
            for j in range(len(spectrum)):
                if window[plane][0] <= spectrum[j][1] <= window[plane][1]:
                    tune = spectrum[j][1]
                    tunes[plane].append(tune)
                    break
            if tune is None:
                return {'H': (None, None), 'V': (None, None)}

    tune_x = np.average(tunes['H']), np.std(tunes['H'])
    tune_y = np.average(tunes['V']), np.std(tunes['V'])
    return {'H': tune_x, 'V': tune_y}


def get_avg_tune_from_harpy(raw_data, start_plateau, end_plateau, variables, qx_window, qy_window, beam, output_path):
    """
        Gets the tune from OMC3's harpy.
        The raw BBQ data is merged and then saved as .sdds for each plateau.
        Each plateau is then analysed by harpy, yielding Q_x,y and their error
    """
    # Create a mask to only get the data in the plateau
    d = raw_data.loc[variables + 'H']
    mask = d['TIMESTAMP'].astype('float') >= start_plateau.timestamp()
    mask = mask & (d['TIMESTAMP'].astype('float') <= end_plateau.timestamp())
    window = {'H': qx_window, 'V': qy_window}

    # Create chunks x seconds long
    seconds_step = 3
    chunks = int((end_plateau - start_plateau).seconds / seconds_step)
    if chunks == 0:
        t = (end_plateau - start_plateau).seconds
        logger.warning(f"The plateau is only {t} seconds long. It will be analyzed in one chunk")
        chunks = 1


    merged_data = {}
    tunes = {'H': [], 'V': []}
    # Get the tune via harpy for each plane
    for plane in ('H', 'V'):
        # Merge the overlapping data returned by Timber
        logger.debug(f"Merging data for beam {beam}, plane {plane}")
        merged_data[plane] = merge_overlap(raw_data.loc[variables + plane]['VALUE'][mask])

    if len(merged_data['H']) != len(merged_data['V']):
        print(len(merged_data['H']))
        print(len(merged_data['V']))
        print()

    for i in range(chunks):
        elements_per_chunk = int(len(merged_data['H']) / chunks)
        data_x = merged_data['H'][elements_per_chunk * i: elements_per_chunk * (i + 1)]
        data_y = merged_data['V'][elements_per_chunk * i: elements_per_chunk * (i + 1)]

        # Save the data as SDDS for each plateau
        filename = f"raw_bbq_B{beam}_{start_plateau}_{end_plateau}_{i:03}.sdds".replace(' ', '_')
        output_sdds = output_path / "sdds"
        output_sdds.mkdir(exist_ok=True)

        if len(data_x) != len(data_y):
            logger.error("Length of the horizontal and vertical data don't match. The SDDS file can not be created.")
            continue

        save_data_to_lhc_sdds(start_plateau.timestamp(),
                              data_x,
                              data_y,
                              "BPM_BBQ",
                              output_sdds / filename)

        # Call Harpy on the chunk!
        model_path = Path("/afs/cern.ch/work/m/mlegarre/public/beta_beat_output/2023-01-26/LHCB1/Models/test_raw_bbq")
        hole_in_one.hole_in_one_entrypoint(
            harpy=True,
            files=[output_sdds / filename],
            accel="lhc",
            model=model_path / 'twiss.dat',
            to_write=['lin', 'spectra', 'full_spectra', 'bpm_summary'],
            unit='m',
            tunes=[.28, .31, 0.],
            nattunes=[.28, .31, 0.],
            clean=False,
            # opposite_direction=opposite_direction,
            sing_val=12,
            outputdir=output_sdds,
            turn_bits=12,
            free_kick=True,
        )
        print("Harpy done")
        tunes['H'].append(tfs.read(output_sdds / f"{filename}.linx").headers['Q1'])
        tunes['V'].append(tfs.read(output_sdds / f"{filename}.liny").headers['Q2'])

    tune_x = np.average(tunes['H']), np.std(tunes['H'])
    tune_y = np.average(tunes['V']), np.std(tunes['V'])
    return {'H': tune_x, 'V': tune_y}

def get_max_peak(x_data, y_data, plane, window):
    # Plot the maximum peaks for each plane
    # Find the peaks via scipy
    peaks, _ = signal.find_peaks(y_data,  # power density data
                                 distance=100,  # minimum distance between peaks
                                 )
    # Get a window to only get the interesting peaks
    tune_window = (x_data[peaks] >= window[plane][0]) & (x_data[peaks] <= window[plane][1])

    # Get the peak amplitude associated to a tune (e.g. 2.6)
    # Sort it in reverse order
    peak_amp, tunes = zip(*sorted(zip(y_data[peaks][tune_window],
                                      x_data[peaks][tune_window]),
                                  reverse=True))
    return tunes[0]


def get_avg_tune_from_spectrogram(f, Sxx, kernel_size, qx_window, qy_window):
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
            tune = get_max_peak(f[plane], data, plane, window)
            tunes[plane].append(tune)
            # print(f'  Segment {i}: {tune}')

        avg_tunes[plane] = [np.mean(tunes[plane]),
                            np.std(tunes[plane])]
        # print(f'Average tune in {plane}: {np.mean(tunes[plane])} +/- {np.std(tunes[plane])}')
    return avg_tunes


def add_points(tune_x, tune_y, i, j, fp, out_tfs, data, qx_window, qy_window, quartiles, plateau_length, bad_tunes,
               method="bbq", raw_bbq_data=None, seconds_step=None, kernel_size=None, beam=None, output_path=None):
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
        tunes_from_raw = get_avg_tune_from_spectrogram(f, Sxx, kernel_size, qx_window, qy_window)
        tune_avg_x, std_x = tunes_from_raw['H']
        tune_avg_y, std_y = tunes_from_raw['V']
    elif method == "raw_bbq_naff":
        start_plateau = isoparse(data['TIME'][fp])
        end_plateau = isoparse(data['TIME'][i - 1])
        variables = f'LHC.BQBBQ.CONTINUOUS_HS.B{beam}:ACQ_DATA_'
        tunes_from_raw = get_avg_tune_from_naff(raw_bbq_data, start_plateau, end_plateau, variables, seconds_step,
                                                qx_window, qy_window)
        tune_avg_x, std_x = tunes_from_raw['H']
        tune_avg_y, std_y = tunes_from_raw['V']
    elif method == "raw_bbq_harpy":
        start_plateau = isoparse(data['TIME'][fp])
        end_plateau = isoparse(data['TIME'][i - 1])
        variables = f'LHC.BQBBQ.CONTINUOUS_HS.B{beam}:ACQ_DATA_'
        tunes_from_raw = get_avg_tune_from_harpy(raw_bbq_data, start_plateau, end_plateau, variables, qx_window, qy_window, beam, output_path)
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
                        bad_tunes, method, raw_bbq_file=None, seconds_step=None, kernel_size=None, beam=None):
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
                                    plateau_length, bad_tunes, method, raw_data, seconds_step, kernel_size, beam, output_path)

            # Reset the counters
            tune_x = []
            tune_y = []
            fp = i

        last_frf = data['F_RF'][i]

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
