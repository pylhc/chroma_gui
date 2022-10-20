import logging
from pathlib import Path
import pandas as pd
import numpy as np
import json
import tfs

RESOURCES = Path(__file__).parent.parent / "resources"


def create_response_matrix_f1004_dq3(observables, chroma_factor, beam):
    kcd_resources = RESOURCES / "normal_decapole"
    kcds = json.load(open(kcd_resources / "strengths.json"))[str(beam)]

    indices = {}
    for i, k in enumerate(kcds.keys()):
        indices[k] = i

    # TODO
    # Copy the full RDT file here
    # Copy the PTC normal files

    # Read the PTC Normal files containing Q'''x and Q'''y
    ptc_files = {k_: tfs.read(kcd_resources / f'ptc_normal_{k}.tfs') for k_ in kcds.keys()}

    # Read the PTC Normal file without any MCD powering, to serve as base
    ptc_base = tfs.read(kcd_resources / f'ptc_normal_NoneB{beam}.tfs')

    # Read the tracking analysis result for f1004
    full_df_f1004 = pd.read_csv(kcd_resources / f'complete_f1004_B{beam}.csv', header=[0, 1], index_col=0)

    rdt = 'f1004_x'
    TO_CORRECT = observables
    FACTOR_CHROMA = chroma_factor  # number of more points to add for the chroma

    # Temporary response matrix
    R_kcd = {}
    for kcd in kcds.keys():
        name = f'KCD.{kcd}'

        # Get the Δ of RDT, compared to the base without any MCD
        real = full_df_f1004.loc[name][f'{rdt} RE'] - full_df_f1004.loc['KCD.NoneB1'][f'{rdt} RE']
        imag = full_df_f1004.loc[name][f'{rdt} IMAG'] - full_df_f1004.loc['KCD.NoneB1'][f'{rdt} IMAG']

        # Same for the chromaticity
        chroma_base = ptc_base[ptc_base['ORDER1'] == 3]
        chroma = ptc_files[kcd][ptc_files[kcd]['ORDER1'] == 3]
        q3x = chroma[chroma['NAME'] == 'DQ1']['VALUE'].values[0] - chroma_base[chroma['NAME'] == 'DQ1']['VALUE'].values[
            0]
        q3y = chroma[chroma['NAME'] == 'DQ2']['VALUE'].values[0] - chroma_base[chroma['NAME'] == 'DQ2']['VALUE'].values[
            0]

        # Sort the index so we're sure to have the values where we want them
        real = real.sort_index()
        imag = imag.sort_index()
        # Form the vector
        obs = []
        # Only add the observables we want to correct
        if 'f1004' in TO_CORRECT:
            obs += list(real.values)
            obs += list(imag.values)
        if 'DQ3' in TO_CORRECT:
            # There are many BPMs for the RDT, to offset that we'll just include several times the chromaticity
            # The number of observables for the RDT is the same as for the chroma
            obs += [q3x] * FACTOR_CHROMA
            obs += [q3y] * FACTOR_CHROMA

        observables = np.array(obs)

        # Once we got the observables, we can construct each part of the matrix
        R_kcd[kcd] = (1 / kcds[kcd]) * observables

    # Create the response matrix which all the previous columns
    R = list()
    for kcd in kcds.keys():
        R.append(R_kcd[kcd])
    R = np.vstack(R).T

    # Correctors to use
    corrs = ['A12B1', 'A23B1', 'A34B1', 'A45B1', 'A56B1', 'A67B1', 'A78B1', 'A81B1']

    # Rebuild the response matrix, taking only those ones into account
    new_R = list()
    for c in corrs:
        new_R.append(R[:, indices[c]])
    new_R = np.vstack(new_R).T

    # Fill the NaN values in the Matrix
    # TODO remove those BPMs in the observed dataframe and the Response Matrix
    new_R[np.isnan(new_R)] = 0

    return new_R
