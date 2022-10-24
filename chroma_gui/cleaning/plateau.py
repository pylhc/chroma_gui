from chroma_gui.cleaning.constants import (
    RF_VARIABLE,
    TUNE_VARS,
    X_VAR_INDICATOR,
    DPP_FILE,
)
import pandas as pd
from datetime import datetime
import tfs


# Read the data to get the RF first
def get_time_data(line):
    timestamp, value = line.split(',')
    timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f')
    value = float(value)
    return timestamp, value


def append(df, dict_to_add):
    new_df = pd.DataFrame([dict_to_add], columns=dict_to_add.keys())
    res_df = pd.concat([df, new_df], ignore_index=True)
    return res_df


def construct_rf_data_from_csv(timber_data, rf_beam):
    rf_data = pd.DataFrame(columns=['TIME', 'F_RF'])
    with open(timber_data) as f:
        freq = 0  # boolean flag to check the variable
        for i, line in enumerate(f):
            if line.startswith('#'):
                continue
            if line.startswith('VARIABLE'):
                freq = 0
                if line[len('VARIABLE: '):].strip() == RF_VARIABLE.format(beam=rf_beam):
                    freq = 1
                continue

            if freq and not line.startswith('Timestamp') and line.strip() != '':
                timestamp, value = get_time_data(line)
                rf_data = append(rf_data, {'TIME': timestamp, 'F_RF': value})
    return rf_data


# The timestamps aren't always equal for each variable
# The idea is to get the last known frequency for a specific timestamp
def get_rf(rf_data, timestamp, nominal_rf):
    mask = rf_data['TIME'] <= timestamp
    rf_before = rf_data.loc[mask]

    if len(rf_before) == 0:  # can happen if data has been redacted
        return nominal_rf

    last_rf = rf_before.iloc[-1]
    return last_rf['F_RF']


# Same thing for the DPP
def get_dpp(dpp_data, timestamp):
    mask = dpp_data['TIME'] <= timestamp
    index = dpp_data.loc[mask].index[-1]
    last_dpp = dpp_data.iloc[index]['DPP']

    # Sometimes the DPP is in the middle of two plateaus, fix that
    dpp_m1 = dpp_data.iloc[index - 1]['DPP']
    dpp_p1 = dpp_data.iloc[index + 1]['DPP']

    if last_dpp != dpp_m1 and last_dpp != dpp_p1:
        return dpp_m1
    else:
        return last_dpp


# Then get the tunes
def get_tunes(timber_data, beam, rf_beam, start_time, end_time, nominal_rf, alpha):
    """
    Builds a dataFrame with the tune, RF and DPP 'chunked' into plateaux
    """
    # Get the RF as a dataframe
    rf_data = construct_rf_data_from_csv(timber_data, rf_beam)

    # If the nominal_rf is None, set it right now as it is the first point of the measurement
    if nominal_rf is None:
        nominal_rf = rf_data['F_RF'].iloc[0]

    data = pd.DataFrame(columns=['TIME', 'F_RF', 'QX', 'QY', 'DPP'])
    with open(timber_data) as f:
        tune = 0  # boolean flag to check the variable
        axis = ''
        for i, line in enumerate(f):
            if line.startswith('#'):
                continue
            if line.startswith('VARIABLE'):
                tune = 0

                beam_tune_vars = [var.format(beam=beam) for var in TUNE_VARS]
                if line[len('VARIABLE: '):].strip() in beam_tune_vars:
                    tune = 1
                    axis = 'X' if X_VAR_INDICATOR in line else 'Y'
                continue

            if tune and not line.startswith('Timestamp') and line.strip() != '':
                timestamp, value = get_time_data(line)

                if timestamp < start_time or timestamp > end_time:
                    continue

                qx = value if axis == 'X' else 0
                qy = value if axis == 'Y' else 0

                # Check if the timestamp exists already
                # If not, add the line
                mask = data['TIME'] == timestamp
                if data[mask].empty:
                    # Get the RF frequency of the point
                    freq = get_rf(rf_data, timestamp, nominal_rf)

                    # Compute the DPP
                    dpp = (-1 / alpha) * (freq - nominal_rf) / nominal_rf

                    # And add the tunes
                    to_add = {'TIME': timestamp, 'F_RF': freq, 'QX': qx, 'QY': qy, 'DPP': dpp}
                    data = append(data, to_add)
                else:
                    i = data[mask].index.to_numpy()[0]
                    if qy:
                        data.at[i, 'QY'] = qy
                    elif qx:
                        data.at[i, 'QX'] = qx

    # Fix types
    new_data = data.astype({'TIME': 'string', 'F_RF': 'float64', 'QX': 'float64', 'QY': 'float64', 'DPP': 'float64'})

    # Create the TFS with its headers
    tfs_data = tfs.TfsDataFrame(new_data)
    tfs_data.headers['ALFA'] = alpha
    tfs_data.headers['F_RF'] = nominal_rf
    tfs_data.headers['BEAM'] = f'B{beam}'

    return tfs_data


def create_plateau(path, timber_data, rf_beam, start_time, end_time, nominal_rf, alpha: dict):
    """
    Wrapper function to get plateaus from B1 and B2 and save them
    """
    plateau_b1 = get_tunes(path / timber_data, 1, rf_beam, start_time, end_time, nominal_rf, alpha['B1'])
    plateau_b2 = get_tunes(path / timber_data, 2, rf_beam, start_time, end_time, nominal_rf, alpha['B2'])

    tfs.write(path / DPP_FILE.format(beam=1), plateau_b1)
    tfs.write(path / DPP_FILE.format(beam=2), plateau_b2)
