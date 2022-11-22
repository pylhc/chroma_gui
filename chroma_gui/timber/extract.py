from fileinput import filename

import pytimber
from datetime import datetime
import os
import logging
import pandas as pd

from chroma_gui.timber.constants import (
    FILENAME,
    BACKUP_FILENAME,
    TIMBER_VARS,
    TIMBER_RAW_VARS,
    FILENAME_PKL,
    BACKUP_FILENAME_PKL
)


def extract_from_timber(variables, start_time, end_time):
    ldb = pytimber.LoggingDB(source="nxcals")

    # Get the data. The resulting dict contains the variables.
    # The values are the timestamp and the value itself
    data = ldb.get(variables, start_time, end_time)
    return data


def save_as_csv(path, start_time, end_time, data):
    # Format the dates
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    start_str = start_time.strftime('%Y-%m-%d_%H-%M-%S')
    end_str = end_time.strftime('%Y-%m-%d_%H-%M-%S')

    backup_name = path / BACKUP_FILENAME.format(now=now)

    with open(backup_name, 'w') as result:
        result.write('# Timber Extraction\n')
        result.write(f'# Extracted on: {now}\n')
        result.write(f'# Start Time: {start_str}\n')
        result.write(f'# End Time  : {end_str}\n')

        for var in data.keys():
            if var in TIMBER_RAW_VARS:  # don't write the huge raw BBQ data to the CSV
                continue
            result.write(f'VARIABLE: {var}\n')
            result.write('Timestamp (LOCAL_TIME),Value\n')
            for ts, val in zip((data[var][0]), list(data[var][1])):
                dt = datetime.fromtimestamp(ts)
                dt_str = dt.strftime('%Y-%m-%d %H:%M:%S.%f')
                result.write(f'{dt_str},{val}\n')
            result.write('\n\n')

    logging.info(f"Timber extracted data saved as {FILENAME}")

    # Make a symlink to TIMBER_DATA.csv
    try:
        os.remove(path / FILENAME)
    except:
        pass
    os.symlink(backup_name, path / FILENAME)


def save_as_pickle(path, data):
    """
    Save the timber data as a dataframe, in a pickle object to preserve everything
    """
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    backup_filename = path / BACKUP_FILENAME_PKL.format(now=now)
    filename = path / FILENAME_PKL

    df = pd.DataFrame.from_dict(data=data, columns=['TIMESTAMP', 'VALUE'], orient='index')
    df.to_pickle(backup_filename)
    logging.info(f"Timber pickle object saved as {FILENAME_PKL}")

    # Make a symlink to TIMBER_RAW_DATA.pkl.gz
    try:
        os.remove(filename)
    except:
        pass
    os.symlink(backup_filename, filename)


def extract_usual_variables(start_time, end_time):
    data = extract_from_timber(TIMBER_VARS, start_time, end_time)
    return data


def extract_raw_variables(start_time, end_time):
    data = extract_from_timber(TIMBER_VARS + TIMBER_RAW_VARS, start_time, end_time)
    return data


def read_variables_from_csv(filename, variables):
    """
    Returns the data of a variable contained in a CSV created by Timber web
    """
    var_flag = False
    values = {}  # dict with variables as keys, then a list of tuple(Timestamp, value)
    with open(filename) as f:
        for i, line in enumerate(f):
            if line.startswith('#'):  # Comments
                continue
            if line.startswith('VARIABLE'):
                var_flag = False
                var_name = line[len('VARIABLE: '):].strip()
                if var_name in variables:  # Variable is found
                    if var_name not in values:
                        values[var_name] = []
                    var_flag = True
                continue

            if var_flag and not line.startswith('Timestamp') and line.strip() != '':
                timestamp, value = line.split(',')
                timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f')
                value = float(value)
                values[var_name].append((timestamp, value))
    return values


def get_variables_names_from_csv(filename):
    """
    Returns all the available variables in the timber extract
    """
    variables = []

    with open(filename) as f:
        for i, line in enumerate(f):
            if line.startswith('#'):  # Comments
                continue
            if line.startswith('VARIABLE'):
                var_name = line[len('VARIABLE: '):].strip()
                variables.append(var_name)
                continue
    return variables

