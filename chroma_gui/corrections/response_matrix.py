"""
Response Matrix
---------------

Functions to create the response matrix to correct chromaticity.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tfs
from numpy.linalg import pinv

RESOURCES = Path(__file__).parent.parent / "resources"

logger = logging.getLogger("response_matrix")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def get_rdt_directory(rdt):
    """
    Returns the name of the directory in the optics analysis for a given RDT
    Example: f1004_x => normal_decapole
    """
    j, k, l, m = [int(r) for r in rdt[1:-2]]
    rdt_type = "normal" if (l + m) % 2 == 0 else "skew"
    orders = dict(
        (
            (1, "dipole"),
            (2, "quadrupole"),
            (3, "sextupole"),
            (4, "octupole"),
            (5, "decapole"),
            (6, "dodecapole"),
            (7, "tetradecapole"),
            (8, "hexadecapole"),
        )
    )
    return f"{rdt_type}_{orders[j + k + l + m]}"


def save_full_rdt_df(model, measurements, names, rdt, output=None):
    """
    Given a list of Optics results directories, will create a single dataframe containing the given `rdt` for each simulation.
    """

    # Create the columns of the dataFrame: f1004 re => BPMs, f1004 img => BPMs
    model = tfs.read(model)
    bpm_names = sorted(list(model["NAME"]))

    rdt_directory = get_rdt_directory(rdt)

    columns = [
        [f"{rdt} RE", f"{rdt} IMAG", f"{rdt} AMP"],
        bpm_names,
    ]  # The MultiIndex allows to "embed" columns
    columns_multi = pd.MultiIndex.from_product(columns, names=[f"{rdt}", "BPMs"])

    # Create the dataframe
    full_df = pd.DataFrame(columns=columns_multi)

    # Add the data to the DF from every simulation
    for i, (kcd_dir, name) in enumerate(zip(measurements, names)):
        rdt_df = tfs.read(kcd_dir / "rdt" / rdt_directory / f"{rdt}.tfs")

        # Get the RDT values, join the model with the outer method to get the missing BPMs
        real = (
            rdt_df[["NAME", "REAL"]]
            .merge(model["NAME"], how="right")
            .set_index("NAME")
            .sort_index()
            .squeeze()
        )
        img = (
            rdt_df[["NAME", "IMAG"]]
            .merge(model["NAME"], how="right")
            .set_index("NAME")
            .sort_index()
            .squeeze()
        )
        amp = (
            rdt_df[["NAME", "AMP"]]
            .merge(model["NAME"], how="right")
            .set_index("NAME")
            .sort_index()
            .squeeze()
        )

        # Add the data with the KCD name as the index
        full_df.loc[name, f"{rdt} RE"] = [real]
        full_df.loc[name, f"{rdt} IMAG"] = [img]
        full_df.loc[name, f"{rdt} AMP"] = [amp]

    # Each data point is a list with only one value, just take the value
    full_df = full_df.applymap(lambda x: x[0])

    if output is not None:
        full_df.to_csv(output, index=True)
    return full_df


class ResponseMatrix:
    def __init__(self, correctors, simulation_path, beam):
        self.correctors = (
            correctors  # List of circuits used as dict {"corrector": "value used as base"}
        )

        # Store the data in two ways:
        #   the original data
        #   data that can be modified via cleaning
        # Simulated observables that will serve as a basis
        # Local observables are dict {"observable": {"corrector": {value}} for every BPM
        # Global observables are dict {"observable": {"corrector": value}}, one value globally
        self.original_simulated_local_observables = {}
        self.original_simulated_global_observables = {}
        self.simulated_local_observables = {}
        self.simulated_global_observables = {}

        # Measurement observables
        # Those dicts follow the same logic as for the simulated ones, except they don't have the inner "corrector" dict
        self.original_measured_local_observables = {}
        self.original_measured_global_observables = {}
        self.measured_local_observables = {}
        self.measured_global_observables = {}

        # Simulation path and beam actually in use
        # Needs to be changed to add new observables
        self.simulation_path = simulation_path
        self.beam = beam

        # Model for RDT BPMs pruning
        self.model_path = None

        # BPMS in the response matrix
        self.bpms_in_use = []

        self.weights = {}  # contains the weight for each global observable

    def set_beam(self, beam):
        self.beam = beam

    def set_simulation_path(self, sim_path):
        self.simulation_path = sim_path

    def _add_simulated_local_observable(self, name, corrector, values):
        """
        Helper function to add simulated local observables to the class
        The structure is as such:
            local_observables[observable] = {corrector1: [values],
                                             corrector2: [values],
                                             ...
                                             }
        """
        if name not in self.original_simulated_local_observables:
            self.original_simulated_local_observables[name] = {}
        self.original_simulated_local_observables[name][corrector] = values

    def _add_measured_local_observable(self, name, values):
        """
        Helper function to add measured local observables to the class
        The structure is as such:
            local_observables[observable] = [values]
        """
        self.original_measured_local_observables[name] = values

    def _add_simulated_global_observable(self, name, corrector, values):
        """
        Helper function to add simulated global observables to the class
        """
        if name not in self.original_simulated_global_observables:
            self.original_simulated_global_observables[name] = {}
        self.original_simulated_global_observables[name][corrector] = values

    def _add_measured_global_observable(self, name, values):
        """
        Helper function to add measured global observables to the class
        """
        self.original_measured_global_observables[name] = values

    def _add_base_rdt_observable(self, rdt, corrector_name="KCD"):
        """
        Helper function to add the simulated RDT to the response matrix
        """
        # Read the tracking analysis result for f1004
        # Generated by the script that ran the simulations
        rdt_df = pd.read_csv(
            self.simulation_path / f"complete_{rdt}_B{self.beam}.csv", header=[0, 1], index_col=0
        )

        # Get the reference corrector without any aditional strength
        reference = [corr for corr in rdt_df.index if "None" in corr][0]
        for corrector in self.correctors.keys():
            # Get the Δ of RDT, compared to the base without any MCD
            real = rdt_df.loc[corrector][f"{rdt} RE"] - rdt_df.loc[reference][f"{rdt} RE"]

            imag = rdt_df.loc[corrector][f"{rdt} IMAG"] - rdt_df.loc[reference][f"{rdt} IMAG"]

            # Sort the index so we're sure to have the values where we want them
            real = real.sort_index()
            imag = imag.sort_index()

            # Add the RDT to the observables dict
            self._add_simulated_local_observable(f"B{self.beam}_{rdt}_re", corrector, real)
            self._add_simulated_local_observable(f"B{self.beam}_{rdt}_imag", corrector, imag)

            # Create the list of BPMs
            bpms = list(rdt_df.loc[corrector][f"{rdt} RE"].index)
            for bpm in bpms:
                if bpm not in self.bpms_in_use:
                    self.bpms_in_use.append(bpm)

    def _add_base_chromaticity_observable(self, order):
        """
        Helper function to add the simulated chromaticity to the response matrix
        """
        # Read the PTC Normal files containing Q'''x and Q'''y
        ptc_files = {
            kcd: tfs.read(self.simulation_path / f"ptc_normal_{kcd.split('.')[-1]}.tfs")
            for kcd in self.correctors.keys()
        }
        # Read the PTC Normal file without any MCD powering, to serve as base
        ptc_base = tfs.read(self.simulation_path / f"ptc_normal_NoneB{self.beam}.tfs")

        for corrector in self.correctors.keys():
            chroma_base = ptc_base[ptc_base["ORDER1"] == order]
            chroma = ptc_files[corrector][ptc_files[corrector]["ORDER1"] == order]
            dqx = (
                chroma[chroma["NAME"] == "DQ1"]["VALUE"].values[0]
                - chroma_base[chroma["NAME"] == "DQ1"]["VALUE"].values[0]
            )
            dqy = (
                chroma[chroma["NAME"] == "DQ2"]["VALUE"].values[0]
                - chroma_base[chroma["NAME"] == "DQ2"]["VALUE"].values[0]
            )

            self._add_simulated_global_observable(f"DQ{order}X", corrector, dqx)
            self._add_simulated_global_observable(f"DQ{order}Y", corrector, dqy)

    def add_rdt_observable(self, measurement, model, rdt, corrector_name="KCD"):
        """
        Adds the given measured RDT and its simulation counterpart to the response matrix
        """
        # Create a dataframe using the model BPMs, containing both measurements
        self.model_path = model
        observed_df = save_full_rdt_df(
            model=model, measurements=[measurement], names=["Measurement"], rdt=rdt
        )

        re_kcd = observed_df.loc["Measurement"][f"{rdt} RE"].sort_index()
        imag_kcd = observed_df.loc["Measurement"][f"{rdt} IMAG"].sort_index()

        self._add_measured_local_observable(f"B{self.beam}_{rdt}_re", re_kcd)
        self._add_measured_local_observable(f"B{self.beam}_{rdt}_imag", imag_kcd)

        # Load the simulated data
        self._add_base_rdt_observable(rdt, corrector_name=corrector_name)

    def add_chromaticity_observable(self, measurement, order, weight):
        """
        Adds the given chromaticity measurement and its simulation counterpart to the response matrix
        """
        # Get the observed chromaticity
        chroma_df = tfs.read(measurement / "chromaticity.tfs")

        mask = chroma_df["BEAM"] == f"B{self.beam}"

        # Errors to correct
        # X Axis
        mask_p = mask & (chroma_df["AXIS"] == "X")
        dq3x_p = chroma_df[mask_p]["Q3"].values[0]

        # Y Axis
        mask_p = mask & (chroma_df["AXIS"] == "Y")
        dq3y_p = chroma_df[mask_p]["Q3"].values[0]

        # There is only one value here, a factor is applied to counterbalance the other observables
        dqx = dq3x_p
        dqy = dq3y_p

        # Add the data to the response matrix
        self._add_measured_global_observable(f"DQ{order}X", dqx)
        self._add_measured_global_observable(f"DQ{order}Y", dqy)

        # Since it is a global observable, a weight needs to be applied to counterbalance other observables
        self.weights[f"DQ{order}X"] = weight
        self.weights[f"DQ{order}Y"] = weight

        # Load the simulated data
        self._add_base_chromaticity_observable(order=3)

    def add_zero_chromaticity_observable(self, order, weight):
        """
        Adds a 0 chromaticity to the observables.
        This can be used to correct some observables but keeping this one constant.
        """
        # Add the dummy measurement
        zero = np.float64(0)
        self._add_measured_global_observable(f"DQ{order}X", zero)
        self._add_measured_global_observable(f"DQ{order}Y", zero)

        # Set the weights
        self.weights[f"DQ{order}X"] = weight
        self.weights[f"DQ{order}Y"] = weight

        # Load the simulated data
        self._add_base_chromaticity_observable(order=3)

    def add_zero_rdt_observable(self, model, rdt, corrector_name):
        """
        Adds the given RDT filled with 0s so it doesn't change
        """
        self.model_path = model

        # Read the simulated RDT to have the right BPMs and basically replace values by 0
        rdt_df = pd.read_csv(
            self.simulation_path / f"complete_{rdt}_B{self.beam}.csv", header=[0, 1], index_col=0
        )
        rdt_df = rdt_df.iloc[0:1]
        rdt_df.index = ["Measurement"]
        for col in rdt_df.columns:
            rdt_df[col].values[:] = 0

        # Then as usual, the "zero" measurement is taken as regular Measurement
        observed_df = rdt_df
        re_kcd = observed_df.loc["Measurement"][f"{rdt} RE"].sort_index()
        imag_kcd = observed_df.loc["Measurement"][f"{rdt} IMAG"].sort_index()
        self._add_measured_local_observable(f"B{self.beam}_{rdt}_re", re_kcd)
        self._add_measured_local_observable(f"B{self.beam}_{rdt}_imag", imag_kcd)

        # Load the simulated data
        self._add_base_rdt_observable(rdt, corrector_name=corrector_name)

    def _clean_local_observables(
        self, inside_arc_number, clean_nan, clean_outliers, clean_IR, quartiles
    ):
        """
        Clean the local observables values:
            - Removes all the indices (BPMs) containing NaN values from all simulations and measurement
            - Remove local outliers BPMs
            - Removes the IR BPMs according to a number, e.g. BPM.(7)R5.B1 < 10 => BPM removed

        The IR BPMs are still removed with ``inside_arc_number`` set to 0. The closest BPMs to the IP will not.
        """
        if clean_nan:
            self._remove_nan_local_observables()

        if clean_outliers:
            self._remove_outlier_bpms(quartiles)

        if clean_IR:
            self._remove_ir_bpms(inside_arc_number)

    def _remove_outlier_bpms(self, quartiles):
        def get_bpms_to_remove(series):
            q1, q3 = np.nanpercentile(series, [quartiles[0], quartiles[1]])  # ignore NaN values
            iqr = q3 - q1
            max_ = q3 + (quartiles[2] * iqr)
            min_ = q1 - (quartiles[2] * iqr)

            # Get the BPMs outside the lower and upper fences
            mask = (series < min_) | (series > max_)
            bpms = list(series[mask].index)
            return bpms

        def remove_bpms(series, bpms):
            # Remove the BPMs from
            df = series.reset_index()
            df = df[~df["BPMs"].isin(bpms)]
            df = df.set_index("BPMs").squeeze()
            return df

        # Iterate a first time to get all the BPMs to remove
        bpms = []
        # Simulated observables
        for observable in self.simulated_local_observables.keys():
            for corrector in self.correctors:
                bpms += get_bpms_to_remove(self.simulated_local_observables[observable][corrector])

        # Measured observables
        for observable in self.measured_local_observables:
            bpms += get_bpms_to_remove(self.measured_local_observables[observable])

        # And then remove those BPMs from all data
        for observable in self.simulated_local_observables.keys():
            for corrector in self.correctors:
                self.simulated_local_observables[observable][corrector] = remove_bpms(
                    self.simulated_local_observables[observable][corrector], bpms
                )
        for observable in self.measured_local_observables:
            self.measured_local_observables[observable] = remove_bpms(
                self.measured_local_observables[observable], bpms
            )

        # Update the BPMs that are still in use
        for bpm in bpms:
            if bpm in self.bpms_in_use:
                self.bpms_in_use.remove(bpm)
        logger.info(f"Removed {len(bpms)} BPMs as outliers")

    def _remove_ir_bpms(self, inside_arc_number):
        """
        Removes the IR BPMs and the BPMs close to it, according to ``inside_arc_number``
        Only the BPMs starting with "BPM." will be kept whatever the number.
            e.g. BPMs such as "BPMSY" will be removed
        """

        def filter_bpm(df):
            df = pd.DataFrame(data=df)
            df = df.reset_index()
            df = df[df["BPMs"].str.startswith("BPM.")]
            mask = df["BPMs"].str.split(".", expand=True)[1]
            mask = mask.str.split("L", expand=True)[0].str.split("R", expand=True)[0]

            df = df[mask.astype(int) >= inside_arc_number]
            df = df.set_index("BPMs").squeeze()

            return df

        # All measurements have the same length and same BPMs.
        # Since we're removing based on BPM name, there is no need to aggregate the names before

        original_len, end_len = 0, 0
        remaining_bpms = []
        # Simulated observables
        for observable in self.simulated_local_observables.keys():
            for corrector in self.correctors:
                original_len = len(self.simulated_local_observables[observable][corrector])
                self.simulated_local_observables[observable][corrector] = filter_bpm(
                    self.simulated_local_observables[observable][corrector]
                )
                end_len = len(self.simulated_local_observables[observable][corrector])

                # Update the remaining BPMS, it will be the same for each observables/corr
                remaining_bpms = list(self.simulated_local_observables[observable][corrector].index)

        # Measured observables
        for observable in self.measured_local_observables:
            self.measured_local_observables[observable] = filter_bpm(
                self.measured_local_observables[observable]
            )

        # Update the bpms still in use
        for bpm in self.bpms_in_use:
            if bpm not in remaining_bpms:
                self.bpms_in_use.remove(bpm)

        logger.info(
            f"Removed {original_len - end_len} IR BPMs from data (BPM.(N). < {inside_arc_number})"
        )

    def _remove_nan_local_observables(self):
        """
        Removes the NaN values from the local observables.
        If at least one NaN is encountered for a BPM, it is then removed from all measurements.
        """
        indices = []  # store all the indices where the value is NaN
        # Iterate over the simulated and measured local observables to get the BPMs indices to remove
        for observable in self.simulated_local_observables.keys():
            for corrector in self.correctors:
                indices += list(
                    np.where(np.isnan(self.simulated_local_observables[observable][corrector]))[0]
                )
        for observable in self.measured_local_observables.keys():
            indices += list(np.where(np.isnan(self.measured_local_observables[observable]))[0])

        indices = list(set(indices))  # remove duplicates

        # Iterate over the observables again, and this time remove the values
        logger.info(f"Removing {len(indices)} BPMs with NaN values from the observables")
        remaining_bpms = []
        for observable in self.simulated_local_observables.keys():
            for corrector in self.correctors:
                self.simulated_local_observables[observable][corrector] = (
                    self.simulated_local_observables[observable][corrector].drop(
                        self.simulated_local_observables[observable][corrector].index[indices]
                    )
                )
                # Update the remaining BPMS, it will be the same for each observables/corr
                remaining_bpms = list(self.simulated_local_observables[observable][corrector].index)

        for observable in self.measured_local_observables.keys():
            self.measured_local_observables[observable] = self.measured_local_observables[
                observable
            ].drop(self.measured_local_observables[observable].index[indices])

        # Update the bpms still in use
        for bpm in self.bpms_in_use:
            if bpm not in remaining_bpms:
                self.bpms_in_use.remove(bpm)

    def _get_response_matrix(self):
        """
        Computes and returns the response matrix, from the local and global observables previously defined
        """
        # temporary response matrix
        tmp_r_matrix = {}

        # Iterate over the simulated observables to create the response matrix
        for corrector in self.correctors.keys():
            values = []

            # Local observables (e.g. RDTs)
            for observable in self.simulated_local_observables.keys():
                values += list(self.simulated_local_observables[observable][corrector])

            # Global (e.g. chromaticity), to be multiplied by a factor
            for observable in self.simulated_global_observables.keys():
                values += [self.simulated_global_observables[observable][corrector]] * self.weights[
                    observable
                ]

            # Divide the observables by the strength of the corrector, and add it to the temporary response matrix
            values = np.array(values)
            tmp_r_matrix[corrector] = (1 / self.correctors[corrector]) * values

        # Create the complete response matrix
        r_matrix = list()
        for corrector in self.correctors.keys():
            r_matrix.append(tmp_r_matrix[corrector])
        r_matrix = np.vstack(r_matrix).T
        logger.info(
            f"Creating a response matrix with {r_matrix.shape[0]} observables "
            f"and {r_matrix.shape[1]} correctors"
        )

        # Replace the NaN values in the Matrix in case there was no cleaning
        r_matrix[np.isnan(r_matrix)] = 0

        self.r_matrix = r_matrix
        return r_matrix

    def _get_measured_array(self):
        """
        Helper function to return a single array containing all the measured observables
        """
        values = []

        # Local observables (e.g. RDTs)
        for observable in self.measured_local_observables.keys():
            values += list(self.measured_local_observables[observable])

        # Global (e.g. chromaticity), to be multiplied by a factor
        for observable in self.measured_global_observables.keys():
            values += [self.measured_global_observables[observable]] * self.weights[observable]

        # Replace the NaN values in the Matrix in case there was no cleaning
        values = np.array(values)
        values[np.isnan(values)] = 0
        return values

    def _copy_original_observables(self):
        # Simulated
        self.simulated_local_observables = deepcopy(self.original_simulated_local_observables)
        self.simulated_global_observables = deepcopy(self.original_simulated_global_observables)

        # Measured
        self.measured_local_observables = deepcopy(self.original_measured_local_observables)
        self.measured_global_observables = deepcopy(self.original_simulated_global_observables)

        # Simulated
        for observable in self.original_simulated_local_observables.keys():
            for corrector in self.correctors:
                self.simulated_local_observables[observable][corrector] = (
                    self.original_simulated_local_observables[observable][corrector].copy()
                )
        for observable in self.original_simulated_global_observables.keys():
            for corrector in self.correctors:
                self.simulated_global_observables[observable][corrector] = (
                    self.original_simulated_global_observables[observable][corrector].copy()
                )

        # Measured
        for observable in self.original_measured_local_observables.keys():
            self.measured_local_observables[observable] = self.original_measured_local_observables[
                observable
            ].copy()
        for observable in self.original_measured_global_observables.keys():
            self.measured_global_observables[observable] = (
                self.original_measured_global_observables[observable].copy()
            )

    def get_corrections(
        self,
        clean_nan=True,
        clean_outliers=True,
        clean_IR=True,
        rcond=0.01,
        inside_arc_number=10,
        quartiles=None,
        decimals_round=0,
    ):
        """
        Computes corrections for the previously given observables.
        """
        if quartiles is None:
            quartiles = [25, 75, 1.5]

        # Start by copying the observables to variables we will be able to remove later on
        self._copy_original_observables()

        # Start by removing the NaN values and the IR BPMs in the observables
        self._clean_local_observables(
            inside_arc_number, clean_nan, clean_outliers, clean_IR, quartiles
        )

        # Get the response matrix from the simulated local and global observables
        r_matrix = self._get_response_matrix()

        # Get the observed values in one array
        measured_values = self._get_measured_array()

        # Compute the correction via pseudo inverse
        inv_r = pinv(r_matrix, rcond=rcond)

        # Get the dot product
        values = inv_r.dot(measured_values)

        corrections = {}
        for key, val in zip(self.correctors, values):
            # Negative because it's a correction
            corrections[f"{key}"] = -round(val, decimals_round)

        return corrections

    def plot_rdt_with_cleaning(self, component, rdt, model, output=None, ylim=None):
        """
        Plots the given RDT component (real, imag, amp) with the cleaned BPMs highlighted
        """

        def format_unit(unit: int):
            num, denum = unit.as_integer_ratio()
            if num == 0:
                unit = ""
            elif denum == "":
                unit = f"{num}"
            elif denum == -1:
                unit = "f{-num}"
            else:
                unit = f"{num}/{denum}"
            return unit

        def get_s_from_bpm(series):
            list_s = []
            list_val = []
            for index, value in series.items():
                s = model[model["NAME"] == index]["S"].values[0]
                list_s.append(s)
                list_val.append(value)
            return zip(*sorted(zip(list_s, list_val)))

        def amp(re, im):
            return (re**2 + im**2) ** 0.5

        def scatter_plot(series, label):
            x, y = get_s_from_bpm(series)
            ax.scatter(x, y, label=label)

        model = tfs.read(model)

        # Get the BPMs that have been cleaned out
        index = self.original_measured_local_observables[
            f"B{self.beam}_{rdt}_re"
        ].index.symmetric_difference(
            self.measured_local_observables[f"B{self.beam}_{rdt}_re"].index
        )

        fig, ax = plt.subplots(figsize=(15, 6))
        if "real" == component:
            print(self.model_path)
            scatter_plot(
                self.measured_local_observables[f"B{self.beam}_{rdt}_re"], label="Real Clean"
            )
            scatter_plot(
                self.original_measured_local_observables[f"B{self.beam}_{rdt}_re"].loc[index],
                label="Removed BPM",
            )
        if "imag" == component:
            scatter_plot(
                self.measured_local_observables[f"B{self.beam}_{rdt}_imag"], label="Imag Clean"
            )
            scatter_plot(
                self.original_measured_local_observables[f"B{self.beam}_{rdt}_imag"].loc[index],
                label="Removed BPM",
            )
        if "amp" == component:
            a = amp(
                self.measured_local_observables[f"B{self.beam}_{rdt}_re"],
                self.measured_local_observables[f"B{self.beam}_{rdt}_imag"],
            )
            scatter_plot(a, label="Amplitude Clean")
            a = amp(
                self.original_measured_local_observables[f"B{self.beam}_{rdt}_re"].loc[index],
                self.original_measured_local_observables[f"B{self.beam}_{rdt}_imag"].loc[index],
            )
            scatter_plot(a, label="Removed BPM")

        # Get the unit of the y axis
        j, k, l, m = [int(e) for e in rdt[1:-2]]  # noqa: E741
        unit_magnet = -(j + k + l + m)  # unit is m^(-order)
        unit = unit_magnet + 1
        unit += (j + k) / 2 + (l + m) / 2
        unit = format_unit(unit)

        ax.legend()
        ax.set_xlabel("s [m]")
        ax.set_ylabel(f"{component.title()} [$m^{{{unit}}}$]")
        if ylim is not None:
            ax.set_ylim(*ylim)

        if output:
            plt.savefig(output)
            logger.info(f"{component.title()} component of {rdt} saved to {output}")
