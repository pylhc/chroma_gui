from datetime import datetime
import typing
import ast
import sys
import json
from json import JSONDecodeError
import tfs
from typing import Tuple, List, Dict
import pyperclip
import matplotlib.pyplot as plt

import logging
from pathlib import Path
from dataclasses import dataclass, field, fields, asdict
import traceback

# PyQt libraries
import qtawesome as qta
from PyQt5.QtGui import QPalette, QStandardItem, QFontMetrics
from PyQt5.QtCore import (
    QDateTime,
    pyqtSignal,
    QThread,
    QAbstractTableModel,
    QModelIndex,
    Qt,
    QEvent,
    QSize,
    pyqtSlot,
)
from PyQt5 import uic
from PyQt5.QtWidgets import (
    QLabel,
    QMainWindow,
    QApplication,
    QDialog,
    QFileDialog,
    QMessageBox,
    QTableView,
    QSizePolicy,
    QHeaderView,
    QComboBox,
    QStyledItemDelegate,
    qApp,
    QListWidgetItem,
    QHBoxLayout,
    QLayout,
)

# Chroma-GUI specific libraries
from chroma_gui import __version__ as chroma_gui_version
import chroma_gui.timber as timber
from chroma_gui.timber import (
    get_variables_names_from_csv,
    read_variables_from_csv,
)
from chroma_gui.plotting.widget import MplWidget, mathTex_to_QPixmap
from chroma_gui.plotting import (
    plot_dpp,
    plot_freq,
    plot_timber,
    plot_chromaticity,
    save_chromaticity_plot,
)
from chroma_gui.cleaning import plateau, clean
from chroma_gui.chromaticity import (
    get_chromaticity,
    construct_chroma_tfs,
    get_maximum_chromaticity,
    get_chromaticity_df_with_notation,
    get_chromaticity_formula
)
import chroma_gui.cleaning.constants as cleaning_constants
from chroma_gui.constants import CHROMA_FILE, RESPONSE_MATRICES, CONFIG, CHROMA_COEFFS
from chroma_gui.corrections import response_matrix

logger = logging.getLogger('chroma_GUI')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

RESOURCES = Path(__file__).parent / "resources"

# Load the different UI windows
new_measurement_class = uic.loadUiType(RESOURCES / "ui_components" / "new_measurement.ui")[0]
main_window_class = uic.loadUiType(RESOURCES / "ui_components" / "chroma_gui.ui")[0]
rcparams_window_class = uic.loadUiType(RESOURCES / "ui_components" / "mpl_rcparams.ui")[0]


class ChromaticityTableModel(QAbstractTableModel):
    def __init__(self, dataframe: tfs.TfsDataFrame, parent=None):
        QAbstractTableModel.__init__(self, parent)
        self._dataframe = dataframe

    def rowCount(self, parent=QModelIndex()) -> int:
        """ Override method from QAbstractTableModel

        Return row count of the pandas DataFrame
        """
        if parent == QModelIndex():
            return len(self._dataframe)

        return 0

    def columnCount(self, parent=QModelIndex()) -> int:
        """Override method from QAbstractTableModel

        Return column count of the pandas DataFrame
        """
        if parent == QModelIndex():
            return len(self._dataframe.columns)
        return 0

    def data(self, index: QModelIndex, role=Qt.ItemDataRole):
        """Override method from QAbstractTableModel

        Return data cell from the pandas DataFrame
        """
        if not index.isValid():
            return None

        if role == Qt.TextAlignmentRole:
            return Qt.AlignRight

        if role == Qt.DisplayRole:
            return str(self._dataframe.iloc[index.row(), index.column()])

        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: Qt.ItemDataRole):
        """Override method from QAbstractTableModel

        Return dataframe index as vertical header data and columns as horizontal header data.
        """
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._dataframe.columns[section])

            if orientation == Qt.Vertical:
                return str(self._dataframe.index[section])

        return None

    def getDataFrame(self):
        return self._dataframe


@dataclass
class Measurement:
    """
    holds measurement specific data such as paths and measurement times
    """
    # Paths for the measurement itself and the two beam models
    path: Path
    model_path: Dict[str, Path] = field(default_factory=lambda: {"B1": None, "B2": None})

    # Metadata about the measurement
    description: str = None

    # Settings for extraction and analysis
    nominal_rf: float = None
    # Extraction times
    start_time: datetime = None
    end_time: datetime = None
    # Analysis times
    cleaning_start_time: datetime = None
    cleaning_end_time: datetime = None

    @classmethod
    def from_folder(cls: typing.Type["Measurement"], path: Path):
        """
        Returns a Measurement object created via the "measurement.info" contained in the given folder
        """
        measurement_info_path = Path(path) / 'measurement.info'
        measurement_info = json.load(open(measurement_info_path))

        measurement = cls(**measurement_info)

        # Fix the datetime and Path types as we're only reading strings from the json
        for field in fields(measurement):
            value = getattr(measurement, field.name)
            if value is None:
                continue
            if field.type is datetime:  # Dates
                setattr(measurement, field.name, datetime.fromisoformat(value))
            elif field.type is Path:  # Paths
                setattr(measurement, field.name, Path(value))
            elif field.type is Dict[str, Path]:  # Dictionaries with Paths
                for key in value.keys():
                    getattr(measurement, field.name)[key] = Path(value[key])

        return measurement

    def get_alpha(self):
        alpha = {"B1": None, "B2": None}
        for beam in alpha.keys():
            twiss = tfs.read(Path(self.model_path[beam]) / 'twiss.dat')
            alpha[beam] = twiss.headers['ALFA']
        return alpha


    def get_timber_status(self):
        """
        Check if a timber extraction data exists and return a small message with details.
        Otherwise, just say there is nothing.
        """
        path_timber_data = Path(self.path / timber.constants.FILENAME)
        if path_timber_data.exists():
            start = None
            end = None
            extracted_on = None
            with open(path_timber_data) as f:
                while True:
                    line = f.readline()
                    if "Start Time" in line:
                        start = line.strip()
                    if "End Time" in line:
                        end = line.strip()
                    if "Extracted on" in line:
                        extracted_on = line.strip()
                    if not line.startswith("#"):
                        break
            message = "Existing extracted data:"
            message += f"\n{extracted_on}\n{start}\n{end}"
            return message, True
        else:
            return "No extraction in directory yet", False

    def get_cleaning_status(self):
        path_dpp = Path(self.path / cleaning_constants.DPP_FILE.format(beam=1))
        path_cleaned = Path(self.path / cleaning_constants.CLEANED_DPP_FILE.format(beam=1))
        return path_dpp.exists(), path_cleaned.exists()

    def get_chroma_status(self):
        path_chroma = Path(self.path / CHROMA_FILE)
        if path_chroma.exists():
            chroma_tfs = tfs.read(path_chroma)
            return True, (chroma_tfs.headers['MIN_FIT_ORDER'], chroma_tfs.headers['MAX_FIT_ORDER'])
        return False, None

    def save_as_json(self):
        """
        Saves the measurement fields as json
        """
        measurement_info_path = Path(self.path) / 'measurement.info'

        # Converts the types to something json can serialize
        data = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if value is None:
                continue

            if field.type is datetime:  # Dates
                data[field.name] = value.isoformat()
            elif field.type is Path:  # Paths
                data[field.name] = str(value)
            elif field.type is Dict[str, Path]:  # Dictionaries with Paths
                data[field.name] = {}
                for key in value.keys():
                    data[field.name][key] = str(value[key])
            else:
                data[field.name] = value

        json.dump(data, open(measurement_info_path, 'w'), indent=4)


class ExternalProgram(QThread):
    finished = pyqtSignal()
    progress = pyqtSignal(float)

    def __init__(self, *args):
        super(QThread, self).__init__()
        self.args = args

        # Base progress is added to the progress bar as the cleaning is done on two independent beams
        self.base_progress = 0
        # Connect the progress signal
        self.progress.connect(self.progress_callback)

    def extractTimber(self):
        # Get the arguments from the creation of the object
        start, end = self.args

        # Tell the user we're extracting
        main_window = findMainWindow()
        main_window.statusTimberExtractionLabel.setText("Extracting data from Timber…")

        # Extract the data from timber
        logger.info("Starting PyTimber extraction")
        data = timber.extract.extract_usual_variables(start, end)

        # Save the data in the measurement path
        measurement_path = main_window.measurement.path
        timber.extract.save_as_csv(measurement_path, start, end, data)

        # If the user tells us to extract the RAW data, save it
        if main_window.rawBBQCheckBox.isChecked():
            data = timber.extract.extract_raw_variables(start, end)
            #timber.extract.save_as_pickle(measurement_path, data)
            timber.extract.save_as_hdf(measurement_path, data)

        logger.info("PyTimber extraction finished")
        self.finished.emit()

    def createPlateaus(self):
        path, filename, rf_beam, start, end, nominal_rf, alpha = self.args
        plateau.create_plateau(path, filename, rf_beam, start_time=start,
                               end_time=end, nominal_rf=nominal_rf, alpha=alpha)
        self.finished.emit()

    def cleanTune(self):
        (input_file_B1, input_file_B2, output_path, output_filename_B1, output_filename_B2, qx_window, qy_window,
         quartiles, plateau_length, bad_tunes) = self.args

        # Reset the progress bar
        self.base_progress = 0
        self.progress_callback(0)

        # Beam 1
        clean.clean_data_for_beam(input_file_B1, output_path, output_filename_B1, qx_window, qy_window, quartiles,
                                  plateau_length, bad_tunes, method="bbq", signal=self.progress)

        # Beam 2
        self.base_progress = 100  # max value is 200
        clean.clean_data_for_beam(input_file_B2, output_path, output_filename_B2, qx_window, qy_window, quartiles,
                                  plateau_length, bad_tunes, method="bbq", signal=self.progress)

        self.progress_callback(100)
        self.finished.emit()

    @pyqtSlot(float)
    def progress_callback(self, progress):
        main_window = findMainWindow()
        main_window.cleaningProgressBar.setValue(self.base_progress + int(progress))

    def cleanTuneRawBBQ(self):
        (input_file, input_file_raw, output_path, output_filename_B1, output_filename_B2, qx_window, qy_window,
         plateau_length, seconds_step, kernel_size, method, bad_tunes) = self.args

        quartiles = None

        # Reset the progress bar
        self.base_progress = 0
        self.progress_callback(0)

        # Beam 1
        clean.clean_data_for_beam(input_file, output_path, output_filename_B1, qx_window, qy_window, quartiles,
                                  plateau_length, bad_tunes, method=method, raw_bbq_file=input_file_raw,
                                  seconds_step=seconds_step, kernel_size=kernel_size, beam=1, signal=self.progress)

        # Beam 2
        self.base_progress = 100
        clean.clean_data_for_beam(input_file, output_path, output_filename_B2, qx_window, qy_window, quartiles,
                                  plateau_length, bad_tunes, method=method, raw_bbq_file=input_file_raw,
                                  seconds_step=seconds_step, kernel_size=kernel_size, beam=2, signal=self.progress)

        self.progress_callback(100)
        self.finished.emit()

    def computeChroma(self):
        input_file_B1, input_file_B2, output_path, fit_orders, dpp_range = self.args

        # Construct the resulting TFS
        chroma_tfs = construct_chroma_tfs(fit_orders)

        # Beam 1
        if input_file_B1 is not None:
            chroma_tfs = get_chromaticity(input_file_B1, chroma_tfs, dpp_range, fit_orders, 'X')
            chroma_tfs = get_chromaticity(input_file_B1, chroma_tfs, dpp_range, fit_orders, 'Y')

        # Beam 2
        if input_file_B2 is not None:
            chroma_tfs = get_chromaticity(input_file_B2, chroma_tfs, dpp_range, fit_orders, 'X')
            chroma_tfs = get_chromaticity(input_file_B2, chroma_tfs, dpp_range, fit_orders, 'Y')

        tfs.write(output_path / CHROMA_FILE, chroma_tfs)

        self.finished.emit()

    def computeCorrections(self):
        optics_paths, measurement_path, method, observables, chroma_factor, rcond, keep_dq3_constant,\
        clean_nan, clean_outliers, clean_IR_number = self.args

        if method == "Global":
            chromaticity_values = tfs.read(measurement_path / CHROMA_FILE)
            coefficients = json.load(open(CHROMA_COEFFS))

            for beam in [1, 2]:
                text = ""
                for obs in observables:
                    order = obs.split("DQ")[1]

                    # Get the measured values
                    mask = chromaticity_values['BEAM'] == f'B{beam}'
                    mask = mask & (chromaticity_values['UP_TO_ORDER'] == chromaticity_values['UP_TO_ORDER'].max())
                    mask_x = mask & (chromaticity_values['AXIS'] == 'X')
                    mask_y = mask & (chromaticity_values['AXIS'] == 'Y')
                    dqx = chromaticity_values[mask_x][f'Q{order}'].values[0]
                    dqy = chromaticity_values[mask_y][f'Q{order}'].values[0]

                    # The chromaticity is simply an affine function that depends on the corrector strength
                    # Get the point where dqx and dqy cross to minimize both planes
                    dq_corr = (dqy - dqx) / (coefficients[str(beam)][order][0] - coefficients[str(beam)][order][1])
                    text = text + f'DQ{order}Corrector.B{beam} = {dq_corr:6.4f} ;\n'

                main_window = findMainWindow()
                main_window.corrections[f'B{beam}'] = text

        elif method == "Local":
            # Compute the corrections for each beam
            for beam in [1, 2]:
                logger.info(f"Computing corrections for Beam {beam}")
                # Get the strengths of the magnets used for simulation
                strengths_mcd = json.load(open(RESOURCES / "normal_decapole" / "strengths.json"))

                # Create the basic response matrix object
                simulations = Path(RESOURCES / "normal_decapole")
                resp = response_matrix.ResponseMatrix(strengths_mcd[str(beam)], simulations, beam=beam)

                # Add the observables
                # Add the RDT to the response matrix
                if "f1004" in observables:
                    optics_path = optics_paths[beam]
                    model_path = RESOURCES / "normal_decapole" / f"twiss_b{beam}.tfs"
                    resp.add_rdt_observable(Path(optics_path), model_path, "f1004_x")

                # Add the Chromaticity to the response matrix
                chroma_path = measurement_path
                if keep_dq3_constant:
                    resp.add_zero_chromaticity_observable(order=3, weight=chroma_factor)
                elif "DQ3" in observables:
                    resp.add_chromaticity_observable(chroma_path, order=3, weight=chroma_factor)

                # Get the corrections
                corrections = resp.get_corrections(rcond=rcond,
                                                   clean_nan=clean_nan,
                                                   clean_outliers=clean_outliers,
                                                   clean_IR=(clean_IR_number != 0),
                                                   inside_arc_number=clean_IR_number
                                                   )

                # Set the text edits with the computed corrections
                text = ""
                for key, val in corrections.items():
                    if val > 0:
                        text += f"{key} := {key} + {val:6d} ;\n"
                    else:
                        text += f"{key} := {key} - {val*-1:6d} ;\n"
                main_window = findMainWindow()
                main_window.corrections[f'B{beam}'] = text

        self.finished.emit()


class CheckableComboBox(QComboBox):
    # Subclass Delegate to increase item height
    class Delegate(QStyledItemDelegate):
        def sizeHint(self, option, index):
            size = super().sizeHint(option, index)
            size.setHeight(20)
            return size

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Make the combo editable to set a custom text, but readonly
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        # Make the lineedit the same color as QPushButton
        palette = qApp.palette()
        palette.setBrush(QPalette.Base, palette.button())
        self.lineEdit().setPalette(palette)

        # Use custom delegate
        self.setItemDelegate(CheckableComboBox.Delegate())

        # Update the text when an item is toggled
        self.model().dataChanged.connect(self.updateText)

        # Hide and show popup when clicking the line edit
        self.lineEdit().installEventFilter(self)
        self.closeOnLineEditClick = False

        # Prevent popup from closing when clicking on an item
        self.view().viewport().installEventFilter(self)

    def resizeEvent(self, event):
        # Recompute text to elide as needed
        self.updateText()
        super().resizeEvent(event)

    def eventFilter(self, object, event):
        if object == self.lineEdit():
            if event.type() == QEvent.MouseButtonRelease:
                if self.closeOnLineEditClick:
                    self.hidePopup()
                else:
                    self.showPopup()
                return True
            return False

        if object == self.view().viewport():
            if event.type() == QEvent.MouseButtonRelease:
                index = self.view().indexAt(event.pos())
                item = self.model().item(index.row())

                if item.checkState() == Qt.Checked:
                    item.setCheckState(Qt.Unchecked)
                else:
                    item.setCheckState(Qt.Checked)
                return True
        return False

    def showPopup(self):
        super().showPopup()
        # When the popup is displayed, a click on the lineedit should close it
        self.closeOnLineEditClick = True

    def hidePopup(self):
        super().hidePopup()
        # Used to prevent immediate reopening when clicking on the lineEdit
        self.startTimer(100)
        # Refresh the display text when closing
        self.updateText()

    def timerEvent(self, event):
        # After timeout, kill timer, and reenable click on line edit
        self.killTimer(event.timerId())
        self.closeOnLineEditClick = False

    def updateText(self):
        texts = []
        for i in range(self.model().rowCount()):
            if self.model().item(i).checkState() == Qt.Checked:
                texts.append(self.model().item(i).text())
        text = ", ".join(texts)

        # Compute elided text (with "...")
        metrics = QFontMetrics(self.lineEdit().font())
        elidedText = metrics.elidedText(text, Qt.ElideRight, self.lineEdit().width())
        self.lineEdit().setText(elidedText)

    def addItem(self, text, data=None):
        item = QStandardItem()
        item.setText(text)
        if data is None:
            item.setData(text)
        else:
            item.setData(data)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
        item.setData(Qt.Unchecked, Qt.CheckStateRole)
        self.model().appendRow(item)

    def addItems(self, texts, datalist=None):
        for i, text in enumerate(texts):
            try:
                data = datalist[i]
            except (TypeError, IndexError):
                data = None
            self.addItem(text, data)

    def currentData(self):
        # Return the list of selected items data
        res = []
        for i in range(self.model().rowCount()):
            if self.model().item(i).checkState() == Qt.Checked:
                res.append(self.model().item(i).data())
        return res


@dataclass
class Config:
    """
    Class for storing user preferences
    """
    # New Measurement Window
    model_path: Path = Path('/user/slops/data/LHC_DATA/OP_DATA/Betabeat/')
    measurements_path: Path = Path('/user/slops/data/LHC_DATA/OP_DATA/Betabeat/')

    # Timber
    extract_raw_timber: bool = False

    # Cleaning
    rf_beam: float = 1
    qx_window: Tuple[float, float] = (0.24, 0.31)
    qy_window: Tuple[float, float] = (0.29, 0.34)
    quartiles: Tuple[float, float] = (0.20, 0.80)
    plateau_length: int = 15
    bad_tune_lines: List[Tuple[float, float]] = field(default_factory=lambda: [(0.2665, 0.2670)])

    plot_dpp: bool = False
    plot_delta_rf: bool = False

    # Matplotlib rcParams
    rcParams: str = None

    @classmethod
    def from_dict(cls: typing.Type["Config"], obj: dict):
        return cls(
            **obj
        )

    def save_field(self, field, data):
        """
        Open the config file and change the given field
        """
        logger.info(f"Saving config field {field}")

        # Read the config
        config_fp = open(CONFIG, 'r+')
        file = json.load(config_fp)
        file[field] = data
        config_fp.close()

        # Write it
        config_fp = open(CONFIG, "w")
        json.dump(file, config_fp, indent=4)


def exceptHook(exc_type, exc_value, exc_tb):
    """
    Function called when an exception occurs
    """
    tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    logger.error(tb)

    # Close the running thread if it caused the crash
    # Otherwise, it might just be the GUI
    main_window = findMainWindow()
    try:
        main_window.thread.terminate()
    except:
        pass

    # Inform the user in the GUI that the function didn't run properly
    stk = traceback.extract_tb(exc_tb, 1)
    function_name = stk[0][2]  # function that caused the issue
    if function_name == "cleanTuneRawBBQ":
        main_window.cleaningStatusLabel.setText("Cleaning failed!")

    return


class MainWindow(QMainWindow, main_window_class):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.setupUi(self)

        # Put the version of the GUI in the title
        self.setWindowTitle(f"Non-Linear Chromaticity GUI — v{chroma_gui_version}")

        self.measurement = None
        self.config = None

        # Thread
        self.thread = None
        self.worker = None

        # Timber variables to display
        self.selectedTimberVariables = []

        # Define the widgets for the plots
        self.plotTimberWidget = None
        self.plotDppB1Widget = None
        self.plotDppB2Widget = None
        self.plotRawTuneB1Widget = None
        self.plotRawTuneB2Widget = None
        self.plotCleanTuneB1Widget = None
        self.plotCleanTuneB2Widget = None
        self.plotChromaB1XWidget = None
        self.plotChromaB1YWidget = None
        self.plotChromaB2XWidget = None
        self.plotChromaB2YWidget = None

        # Define table models for chromaticity
        self.chromaB1TableModel = None
        self.chromaB2TableModel = None

        # Load preferences for file structure
        self.loadConfig()
        self.applyMplStyle()

        # Disable tabs for now, as no measurement has been created or opened yet
        self.enableTimberTab(False)
        self.enableCleaningTab(False)
        self.enableChromaticityTab(False)
        self.enableCorrectionsTab(False)

        self.available_observables = {}
        self.corrections = {"B1": None, "B2": None}
        self.setCorrectionComboBox()

        # Set the info icons on the labels
        self.setInfoIcons()

        # R2 scores for each chromaticity fit
        self.r2scores = {"B1": {"X": 0, "Y": 0},
                         "B2": {"X": 0, "Y": 0}}

    def setInfoIcons(self):
        """
        Iterate through all the labels in the class that have a tooltip, and place a proper info icon next to it
        """
        for name, obj in vars(self).items():
            if type(obj) == QLabel:
                text = obj.text()
                tooltip = obj.toolTip()
                if tooltip.strip() != "":
                    # Create a H layout that will old the text / icon
                    layout = QHBoxLayout()
                    layout.setContentsMargins(0, 0, 0, 0)

                    info_icon_id = "fa5s.question-circle"
                    icon = QLabel()
                    icon.setPixmap(qta.icon(info_icon_id).pixmap(QSize(16, 16)))

                    layout.addWidget(QLabel(text), 0, Qt.AlignLeft)
                    layout.addSpacing(1)
                    layout.addWidget(icon, 0, Qt.AlignLeft)

                    # Set the original tool tip to the icon
                    obj.setToolTip("")
                    icon.setToolTip(tooltip)

                    obj.setText("")  # remove the old text
                    obj.setLayout(layout)  # and replace by the new layout
                    layout.setSizeConstraint(QLayout.SetMinimumSize)
                    layout.setAlignment(Qt.AlignLeft)
                    obj.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

    def applyMplStyle(self):
        if self.config.rcParams is None:
            plt.style.use(RESOURCES / "chroma_gui.mplstyle")
        else:
            for line in self.config.rcParams.split('\n'):
                if not line.startswith("#") and line.strip() != "":
                    key, value = [e.strip() for e in line.split(':')]
                    plt.style.use({key: value})

    def loadConfig(self):
        if CONFIG.exists():
            config_dict = json.load(open(CONFIG))
        else:
            config_dict = {}
        self.config = Config.from_dict(config_dict)

        # Fix the value types
        self.config.measurements_path = Path(self.config.measurements_path)
        self.config.model_path = Path(self.config.model_path)
        for i, tune_window in enumerate(self.config.bad_tune_lines):
            self.config.bad_tune_lines[i] = tuple(tune_window)

        # Set the different line edits
        self.qxWindowLow.setValue(self.config.qx_window[0])
        self.qxWindowHigh.setValue(self.config.qx_window[1])
        self.qyWindowLow.setValue(self.config.qy_window[0])
        self.qyWindowHigh.setValue(self.config.qy_window[1])
        self.q1Quartile.setValue(self.config.quartiles[0])
        self.q3Quartile.setValue(self.config.quartiles[1])
        self.rfBeamComboBox.setCurrentIndex(self.config.rf_beam - 1)  # Index starts at 0: Beam - 1 = index
        self.plateauLength.setValue(self.config.plateau_length)
        self.badTunesLineEdit.setText(str(self.config.bad_tune_lines))

        self.rawBBQCheckBox.setChecked(self.config.extract_raw_timber)
        self.showDppCheckBox.setChecked(self.config.plot_dpp)
        self.showDeltaRfCheckBox.setChecked(self.config.plot_delta_rf)

    def updateLineEdits(self):
        # Update the labels in the Main Window
        alpha = self.measurement.get_alpha()
        self.alfaB1LineEdit.setText(str(alpha['B1']))
        self.alfaB2LineEdit.setText(str(alpha['B2']))
        self.nominalRfLineEdit.setText(str(self.measurement.nominal_rf))
        self.descriptionPlainTextEdit.setPlainText(self.measurement.description)

        # Set the extraction dates via Qt objects
        start = QDateTime.fromString(self.measurement.start_time.strftime("%Y-%m-%dT%H:%M:%S"), 'yyyy-MM-ddThh:mm:ss')
        end = QDateTime.fromString(self.measurement.end_time.strftime("%Y-%m-%dT%H:%M:%S"), 'yyyy-MM-ddThh:mm:ss')
        self.startTimeTimberEdit.setDateTime(start)
        self.endTimeTimberEdit.setDateTime(end)

        # Write if an extraction already exists with some details about the date
        message, extracted = self.measurement.get_timber_status()
        self.statusTimberExtractionLabel.setText(message)

        # Enable the Timber tab for the first time
        self.enableTimberTab(True)

        # Enable the cleaning tab if we've got extracted data
        if extracted:
            self.updateTimberPlot(self.measurement)
            self.updateTimberTable(self.measurement)
            self.enableCleaningTab(True)
            # Get the times from the measurement.info, or set them to the Timber extraction times
            cleaning_start = QDateTime.fromString(self.measurement.cleaning_start_time.strftime("%Y-%m-%dT%H:%M:%S"),
                                                  'yyyy-MM-ddThh:mm:ss') if self.measurement.cleaning_start_time \
                                                                         else start
            cleaning_end = QDateTime.fromString(self.measurement.cleaning_end_time.strftime("%Y-%m-%dT%H:%M:%S"),
                                                'yyyy-MM-ddThh:mm:ss') if self.measurement.cleaning_end_time \
                                                                       else end
            self.startPlateauDateTimeEdit.setDateTime(cleaning_start)
            self.endPlateauDateTimeEdit.setDateTime(cleaning_end)

        # Check if we got cleaned data already
        dpp, cleaned = self.measurement.get_cleaning_status()
        if dpp:  # If we have created plateaus, plot them
            self.plateauFinished(self.measurement)
        else:  # If we don't have plateaus, there is no need to display the cleaning yet
            self.cleaningWidget.setEnabled(False)

        if cleaned:  # the cleaned data exist, it can be plotted!
            self.cleaningFinished(self.measurement)
            self.enableChromaticityTab(True)

        chroma_status, chroma_orders = self.measurement.get_chroma_status()
        if chroma_status:
            self.setChromaticityOrders(chroma_orders)
            self.chromaFinished(self.measurement)

    def setCorrectionComboBox(self):
        """
        This function replaces the ComboBox containing the observables by one with items that can be clicked
        """
        # Remove the existing widget
        self.verticalCorrectionLayout.removeWidget(self.observablesCorrectionComboBox)
        self.observablesCorrectionComboBox.close()

        # Create a new custom ComboBox and add it to the layout
        self.observablesCorrectionComboBox = CheckableComboBox(self)
        self.verticalCorrectionLayout.addWidget(self.observablesCorrectionComboBox)
        self.verticalCorrectionLayout.update()

        # Display the available correction methods
        self.available_observables = json.load(open(RESPONSE_MATRICES))['AVAILABLE_OBSERVABLES']
        self.correctionMethodComboBox.addItems(self.available_observables.keys())

    def correctionMethodComboBoxChanged(self, method: str):
        """
        Set the available observables for the selected method.
        Greys out the unavailable options
        """
        selected_method = self.correctionMethodComboBox.currentText()

        # Set the available observables
        self.observablesCorrectionComboBox.clear()
        self.observablesCorrectionComboBox.addItems(self.available_observables[selected_method])

        # Greys out the options of the method is global
        self.measurementCorrectionB1LineEdit.setEnabled(selected_method == "Local")
        self.measurementCorrectionB2LineEdit.setEnabled(selected_method == "Local")
        self.factorChromaSpinBox.setEnabled(selected_method == "Local")
        self.rcondCorrectionSpinBox.setEnabled(selected_method == "Local")
        self.keepDQ3ConstantcheckBox.setEnabled(selected_method == "Local")
        self.cleanNaNCheckBox.setEnabled(selected_method == "Local")
        self.cleanOutliersCheckBox.setEnabled(selected_method == "Local")
        self.cleanIRSpinBox.setEnabled(selected_method == "Local")

    def enableTimberTab(self, value):
        self.timberTab.setEnabled(value)

    def enableCleaningTab(self, value):
        self.cleaningTab.setEnabled(value)

    def enableChromaticityTab(self, value):
        self.chromaticityTab.setEnabled(value)

    def enableCorrectionsTab(self, value):
        self.correctionsTab.setEnabled(value)

    def newMeasurementClicked(self):
        measurement_dialog = NewMeasurementDialog(self)
        measurement_dialog.show()

    def openMeasurementClicked(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Measurement Directory",
            str(self.config.measurements_path),
        )
        if not folder:
            QMessageBox.warning(self,
                                "Failed to open directory",
                                f"The directory '{folder}' could not be opened")
            return False

        # Try to open the measurement information file
        try:
            self.measurement = Measurement.from_folder(Path(folder))
        except OSError as e:
            QMessageBox.warning(self,
                                "Failed to open measurement",
                                f"{str(e)}")
            logger.error(e)
        except JSONDecodeError as e:
            QMessageBox.warning(self,
                                "Failed to open measurement",
                                f"The file 'measurement.info' is not a valid JSON file")
            logger.error(e)
        except KeyError as e:
            QMessageBox.warning(self,
                                "Failed to open measurement",
                                f"The file 'measurement.info' does not contain the required keys")
            logger.error(e)

        if self.measurement is not None:
            self.updateLineEdits()

    def saveSettingsClicked(self):
        """
        Save the settings into the measurement file
        """
        # Check if a measurement has been opened or created already, otherwise why would we save something?
        if not self.measurement:
            logger.warning("No measurement has been opened or created, the settings can't be saved.")
            return

        # Nominal RF
        nominal_rf = self.nominalRfLineEdit.text()
        if nominal_rf.strip() == "0" or nominal_rf.strip() == 'None':
            nominal_rf = None
        if nominal_rf is not None:
            nominal_rf = float(nominal_rf)
        self.measurement.nominal_rf = nominal_rf

        # Description
        self.measurement.description = self.descriptionPlainTextEdit.toPlainText()

        # Save the measurement
        self.measurement.save_as_json()
        logger.info('Settings saved!')

    def startThread(self, main_function, finish_function, *args):
        """
        Simple wrapper to start a thread
        Arguments:
            - main_function: method of the class `ExternalProgram` to be started as main function of the thread
            - finish_function: method of the class  `MainWindow` to be called when the thread has finished
            - *args: arguments to be passed to the instantiation of the `ExternalProgram` class, those are the
            `main_function` arguments
        """
        # Check if we've got a thread already running
        try:
            if self.thread is not None and self.thread.isRunning():
                logger.warning("A thread is already running, please wait for it to finish")
                return
        except RuntimeError:  # The thread has been deleted, it's all good
            pass

        # Create the thread and the class with our functions
        self.thread = QThread()
        self.worker = ExternalProgram(*args)

        # Move worker to the thread
        self.worker.moveToThread(self.thread)

        # Connect signals and slots
        self.thread.started.connect(getattr(self.worker, main_function))
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(getattr(self, finish_function))
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Start the thread
        self.thread.start()

    def extractTimberClicked(self):
        start = self.startTimeTimberEdit.dateTime().toPyDateTime()
        end = self.endTimeTimberEdit.dateTime().toPyDateTime()

        # Set the start and end times to the Measurement object, and save it
        self.measurement.start_time = start
        self.measurement.end_time = end
        self.measurement.save_as_json()

        # Start the extraction
        logger.info("Starting Timber extraction")
        self.startThread("extractTimber", "timberExtractionFinished", start, end)

    def createPlateausClicked(self):
        start = self.startPlateauDateTimeEdit.dateTime().toPyDateTime()
        end = self.endPlateauDateTimeEdit.dateTime().toPyDateTime()
        rf_beam = int(self.rfBeamComboBox.currentText())
        nominal_rf = self.nominalRfLineEdit.text()

        # Check of the nominal RF exists
        # If it is 0 or None, the plateau creation will take the first value in the dataFrame
        if nominal_rf.strip() == "0" or nominal_rf.strip() == 'None':
            nominal_rf = None
            msg = "The nominal frequency is not set. The first point of the data extracted will be taken. "
            msg += "Be sure that this point is the expected one!"
            logger.warning(msg)
        if nominal_rf is not None:
            nominal_rf = float(nominal_rf)

        # Set the analysis start and end time in the measurement.info
        self.measurement.cleaning_start_time = start
        self.measurement.cleaning_end_time = end
        self.measurement.save_as_json()

        # Start the plateau creation
        logger.info("Starting Plateau Creation")
        self.startThread("createPlateaus", "plateauFinished", self.measurement.path, timber.constants.FILENAME,
                         rf_beam, start, end, nominal_rf, self.measurement.get_alpha())

    def cleanDataClicked(self):
        # Get values from the GUI
        # Tune Window
        qx_low = self.qxWindowLow.value()
        qx_high = self.qxWindowHigh.value()
        qy_low = self.qyWindowLow.value()
        qy_high = self.qyWindowHigh.value()

        # Quartiles
        q1_quartile = self.q1Quartile.value()
        q3_quartile = self.q3Quartile.value()

        # Minimum Plateau Length
        plateau_length = self.plateauLength.value()

        # Raw BBQ specific options
        seconds_step = self.secondsStep.value()
        kernel_size = self.kernelSize.value()

        # Bad tune lines: list of tuples, e.g. [(0.26, 0.28), ]
        try:
            bad_tunes = ast.literal_eval(self.badTunesLineEdit.text())
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            bad_tunes = []
            logger.error("Could not load the bad tunes line. Defaulting to none.")

        # Set the cleaning label
        self.cleaningStatusLabel.setText("Cleaning in progress…")

        logger.info("Starting Tune Cleaning")
        if not self.useRawBBQCheckBox.isChecked():  # classic BBQ
            self.startThread("cleanTune",
                             "cleaningFinished",
                             self.measurement.path / cleaning_constants.DPP_FILE.format(beam=1),
                             self.measurement.path / cleaning_constants.DPP_FILE.format(beam=2),
                             self.measurement.path,
                             cleaning_constants.CLEANED_DPP_FILE.format(beam=1),
                             cleaning_constants.CLEANED_DPP_FILE.format(beam=2),
                             (qx_low, qx_high),
                             (qy_low, qy_high),
                             (q1_quartile, q3_quartile),
                             plateau_length,
                             bad_tunes)
        else:  # raw BBQ
            # Select the method to be called for processing the raw BBQ
            method = "raw_bbq_spectrogram"
            if self.useNAFFCheckBox.isChecked():
                method = "raw_bbq_naff"
            self.startThread("cleanTuneRawBBQ",
                             "cleaningFinished",
                             self.measurement.path / cleaning_constants.DPP_FILE.format(beam=1),
                             self.measurement.path / timber.constants.FILENAME_HDF,
                             self.measurement.path,
                             cleaning_constants.CLEANED_DPP_FILE.format(beam=1),
                             cleaning_constants.CLEANED_DPP_FILE.format(beam=2),
                             (qx_low, qx_high),
                             (qy_low, qy_high),
                             plateau_length,
                             int(seconds_step),
                             int(kernel_size),
                             method,
                             bad_tunes,
                             )

    def plateauFinished(self, measurement=None):
        logger.info("Plateaus done!")
        # This function is called during the Measurement object init when loading a directory
        # This means the 'measurement' object isn't assigned yet to the main window, i.e. 'self.measurement' is None
        if not measurement:
            measurement = self.measurement
        self.updateDppPlot(measurement)
        self.updateRawTunePlot(measurement)
        self.cleaningWidget.setEnabled(True)

    def cleaningFinished(self, measurement=None):
        logger.info("Cleaning done!")
        self.cleaningStatusLabel.setText("Cleaning done")
        self.enableChromaticityTab(True)
        if not measurement:
            measurement = self.measurement
        self.updateCleanedTunePlot(measurement)

    def timberExtractionFinished(self):
        # Update some UI
        self.updateLineEdits()

    def updateDppPlot(self, measurement):
        """
        Add a plot to show the DPP / RF measurement
        """
        # Remove the existing plots, if any
        if self.plotDppB1Widget is not None:
            self.deletePlot(self.plotDppB1Widget)
            self.deletePlot(self.plotDppB2Widget)

        # Create the Matplotlib widgets
        self.plotDppB1Widget = MplWidget()
        self.plotDppB2Widget = MplWidget()

        # Add the widgets to the layout
        self.cleaningPlotB1Layout.addWidget(self.plotDppB1Widget)
        self.cleaningPlotB2Layout.addWidget(self.plotDppB2Widget)

        # Beam 1
        file_path = measurement.path / cleaning_constants.DPP_FILE.format(beam=1)
        plot_dpp(self.plotDppB1Widget.canvas.fig, self.plotDppB1Widget.canvas.ax, file_path)
        self.plotDppB1Widget.canvas.draw()
        self.plotDppB1Widget.show()

        # Beam 2
        file_path = measurement.path / cleaning_constants.DPP_FILE.format(beam=2)
        plot_dpp(self.plotDppB2Widget.canvas.fig, self.plotDppB2Widget.canvas.ax, file_path)
        self.plotDppB2Widget.canvas.draw()
        self.plotDppB2Widget.show()

    def updateRawTunePlot(self, measurement):
        # Add a plot to show the tune before cleaning
        if self.plotRawTuneB1Widget is not None:
            self.deletePlot(self.plotRawTuneB1Widget)
            self.deletePlot(self.plotRawTuneB2Widget)

        self.plotRawTuneB1Widget = MplWidget()
        self.plotRawTuneB2Widget = MplWidget()
        self.rawPlotB1Layout.addWidget(self.plotRawTuneB1Widget)
        self.rawPlotB2Layout.addWidget(self.plotRawTuneB2Widget)

        # Options
        dpp_flag = self.showDppCheckBox.isChecked()
        delta_rf_flag = self.showDeltaRfCheckBox.isChecked()

        # Beam 1
        filepath = measurement.path / cleaning_constants.DPP_FILE.format(beam=1)
        plot_freq(self.plotRawTuneB1Widget.canvas.fig, self.plotRawTuneB1Widget.canvas.ax, filepath,
                  'Raw Tune Measurement for Beam 1', dpp_flag=dpp_flag, delta_rf_flag=delta_rf_flag)
        self.plotRawTuneB1Widget.canvas.draw()
        self.plotRawTuneB1Widget.show()

        # Beam 2
        filepath = measurement.path / cleaning_constants.DPP_FILE.format(beam=2)
        plot_freq(self.plotRawTuneB2Widget.canvas.fig, self.plotRawTuneB2Widget.canvas.ax, filepath,
                  f'Raw Tune Measurement for Beam 2', dpp_flag=dpp_flag, delta_rf_flag=delta_rf_flag)
        self.plotRawTuneB2Widget.canvas.draw()
        self.plotRawTuneB2Widget.show()

    def updateCleanedTunePlot(self, measurement):
        logger.info("Plotting cleaned tune")
        # Add a plot to show the tune before cleaning
        if self.plotCleanTuneB1Widget is not None:
            self.deletePlot(self.plotCleanTuneB1Widget)
            self.deletePlot(self.plotCleanTuneB2Widget)

        self.plotCleanTuneB1Widget = MplWidget()
        self.plotCleanTuneB2Widget = MplWidget()
        self.cleanPlotB1Layout.addWidget(self.plotCleanTuneB1Widget)
        self.cleanPlotB2Layout.addWidget(self.plotCleanTuneB2Widget)

        # Options
        dpp_flag = self.showDppCheckBox.isChecked()
        delta_rf_flag = self.showDeltaRfCheckBox.isChecked()

        # Beam 1
        filepath = measurement.path / cleaning_constants.CLEANED_DPP_FILE.format(beam=1)
        plot_freq(self.plotCleanTuneB1Widget.canvas.fig, self.plotCleanTuneB1Widget.canvas.ax, filepath,
                  'Cleaned Tune Measurement for Beam 1', dpp_flag=dpp_flag, delta_rf_flag=delta_rf_flag,
                  plot_style="line")
        self.plotCleanTuneB1Widget.canvas.draw()
        self.plotCleanTuneB1Widget.show()

        # Beam 2
        filepath = measurement.path / cleaning_constants.CLEANED_DPP_FILE.format(beam=2)
        plot_freq(self.plotCleanTuneB2Widget.canvas.fig, self.plotCleanTuneB2Widget.canvas.ax, filepath,
                  f'Cleaned Tune Measurement for Beam 2', dpp_flag=dpp_flag, delta_rf_flag=delta_rf_flag,
                  plot_style="line")
        self.plotCleanTuneB2Widget.canvas.draw()
        self.plotCleanTuneB2Widget.show()

    def deletePlot(self, widget):
        # Remove widget from the layout and delete the widget
        widget.setParent(None)
        del widget

    def rePlotCleaningClicked(self):
        """
        This buttons allows to replot the plots in the cleaning tab without redoing the analysis
        """
        self.updateRawTunePlot(self.measurement)
        self.updateCleanedTunePlot(self.measurement)

    def getChromaticityOrders(self):
        """
        Returns a list containing the checked chromaticity orders
        """
        checked = []
        for order in range(3, 8):
            dq = getattr(self, f'ChromaOrder{order}CheckBox').isChecked()
            if dq:
                checked.append(order)
        return checked

    def computeChromaClicked(self):
        # Get values from the GUI
        dpp_range_low = self.dppRangeLowSpinBox.value()
        dpp_range_high = self.dppRangeHighSpinBox.value()

        input_file_B1 = self.measurement.path / cleaning_constants.CLEANED_DPP_FILE.format(beam=1)
        input_file_B2 = self.measurement.path / cleaning_constants.CLEANED_DPP_FILE.format(beam=2)
        output_path = self.measurement.path
        fit_orders = self.getChromaticityOrders()
        dpp_range = (dpp_range_low, dpp_range_high)

        # Set the files to None if that beam didn't get a measurement
        if not input_file_B1.exists():
            input_file_B1 = None
        if not input_file_B2.exists():
            input_file_B2 = None

        logger.info("Starting Chromaticity Computing")
        self.startThread("computeChroma", "chromaFinished",
                         input_file_B1, input_file_B2, output_path, fit_orders, dpp_range)
        return

    def chromaFinished(self, measurement=None):
        logger.info('Chromaticity finished computing')
        if not measurement:
            measurement = self.measurement
        self.updateChromaTables(measurement)
        self.updateChromaPlots(measurement)
        self.updateR2scores()
        self.enableCorrectionsTab(True)

    def updateR2scores(self):
        # Set the r2 scores
        current_beam = self.beamChromaticityTabWidget.currentIndex() + 1  # index starts at 0
        self.r2_x_value.setText("{:.5f}".format(round(self.r2scores[f"B{current_beam}"]["X"], 5)))
        self.r2_y_value.setText("{:.5f}".format(round(self.r2scores[f"B{current_beam}"]["Y"], 5)))

    def updateChromaTablesForBeam(self, current_beam):
        current_beam += 1  # index starts at 0
        self.beamChromaticityTableView.setModel(getattr(self, f"chromaB{current_beam}TableModel"))
        self.updateR2scores()

    def updateChromaTables(self, measurement):
        chroma_tfs = tfs.read(measurement.path / CHROMA_FILE)
        chroma_tfs = get_maximum_chromaticity(chroma_tfs)
        chroma_tfs = get_chromaticity_df_with_notation(chroma_tfs)

        # Update the Chromaticity Formula at the top of the table
        order = max(self.getChromaticityOrders())
        latex_formula = get_chromaticity_formula(order)
        pixmap = mathTex_to_QPixmap(latex_formula, fs=12)
        self.chromaticityFormulaLabel.setPixmap(pixmap)

        # Beam 1 and Beam 2 models
        self.chromaB1TableModel = ChromaticityTableModel(chroma_tfs[chroma_tfs['BEAM'] == 'B1'].drop('BEAM', axis=1))
        self.chromaB2TableModel = ChromaticityTableModel(chroma_tfs[chroma_tfs['BEAM'] == 'B2'].drop('BEAM', axis=1))

        # Set the model of the beam depending on the tab selected
        current_beam = self.beamChromaticityTabWidget.currentIndex() + 1  # index starts at 0
        self.beamChromaticityTableView.setModel(getattr(self, f"chromaB{current_beam}TableModel"))
        # Hide the indices
        self.beamChromaticityTableView.verticalHeader().setVisible(False)
        # Select an item when clicking on a cell, not the row
        self.beamChromaticityTableView.setSelectionBehavior(QTableView.SelectItems)
        # Take all the horizontal space
        self.beamChromaticityTableView.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def updateChromaPlots(self, measurement):
        """
        Add a plot to show the Chromaticity for both axes and both beams
        """
        # Remove the existing plots, if any
        if self.plotChromaB1XWidget is not None:
            self.deletePlot(self.plotChromaB1XWidget)
            self.deletePlot(self.plotChromaB1YWidget)
            self.deletePlot(self.plotChromaB2XWidget)
            self.deletePlot(self.plotChromaB2YWidget)

        # Create the Matplotlib widgets
        self.plotChromaB1XWidget = MplWidget()
        self.plotChromaB1YWidget = MplWidget()
        self.plotChromaB2XWidget = MplWidget()
        self.plotChromaB2YWidget = MplWidget()

        size = (10, 10)
        self.plotChromaB1XWidget.canvas.fig.set_size_inches(size)
        self.plotChromaB1YWidget.canvas.fig.set_size_inches(size)
        self.plotChromaB2XWidget.canvas.fig.set_size_inches(size)
        self.plotChromaB2YWidget.canvas.fig.set_size_inches(size)

        # Add the widgets to the layout
        self.beam1ChromaLayout.addWidget(self.plotChromaB1XWidget)
        self.beam1ChromaLayout.addWidget(self.plotChromaB1YWidget)
        self.beam2ChromaLayout.addWidget(self.plotChromaB2XWidget)
        self.beam2ChromaLayout.addWidget(self.plotChromaB2YWidget)

        # General values
        chroma_tfs_file = tfs.read(measurement.path / CHROMA_FILE)

        # Filter the chromaticity orders in case more are checked than already measured
        fit_orders = self.getChromaticityOrders()

        # Beam 1
        dpp_file_b1 = measurement.path / cleaning_constants.CLEANED_DPP_FILE.format(beam=1)
        self.r2scores['B1']['X'] = plot_chromaticity(self.plotChromaB1XWidget.canvas.fig,
                                                     self.plotChromaB1XWidget.canvas.ax,
                                                     dpp_file_b1, chroma_tfs_file, 'X', fit_orders, "B1")
        self.r2scores['B1']['Y'] = plot_chromaticity(self.plotChromaB1YWidget.canvas.fig,
                                                     self.plotChromaB1YWidget.canvas.ax,
                                                     dpp_file_b1, chroma_tfs_file, 'Y', fit_orders, "B1")

        # Beam 2
        dpp_file_b2 = measurement.path / cleaning_constants.CLEANED_DPP_FILE.format(beam=2)
        self.r2scores['B2']['X'] = plot_chromaticity(self.plotChromaB2XWidget.canvas.fig,
                                                     self.plotChromaB2XWidget.canvas.ax,
                                                     dpp_file_b2, chroma_tfs_file, 'X', fit_orders, "B2")
        self.r2scores['B2']['Y'] = plot_chromaticity(self.plotChromaB2YWidget.canvas.fig,
                                                     self.plotChromaB2YWidget.canvas.ax,
                                                     dpp_file_b2, chroma_tfs_file, 'Y', fit_orders, "B2")

        # Set the r2 scores
        self.updateR2scores()

    def savePlotsClicked(self):
        path = self.measurement.path / "plots"
        path.mkdir(exist_ok=True)

        save_chromaticity_plot(self.plotChromaB1XWidget.canvas.fig, path / "Beam1_Qx", formats=['png', 'pdf'])
        save_chromaticity_plot(self.plotChromaB1YWidget.canvas.fig, path / "Beam1_Qy", formats=['png', 'pdf'])
        save_chromaticity_plot(self.plotChromaB2XWidget.canvas.fig, path / "Beam2_Qx", formats=['png', 'pdf'])
        save_chromaticity_plot(self.plotChromaB2YWidget.canvas.fig, path / "Beam2_Qy", formats=['png', 'pdf'])
        logger.info(f"Saved Chromaticity plots to {path}")

    def copyTableClicked(self):
        format_table = self.copyTableComboBox.currentText()
        current_beam = self.beamChromaticityTabWidget.currentIndex() + 1  # index starts at 0
        df = getattr(self, f"chromaB{current_beam}TableModel").getDataFrame()

        df_text = ""
        if format_table == "Markdown":
            df_text = df.to_markdown(index=False)
        elif format_table == "LaTeX":
            df_text = df.to_latex(index=False)

        pyperclip.copy(df_text)
        logger.info(f"Chromaticity Table for beam {current_beam} copied to clipboard.")

    def useRawBBQCheckBoxClicked(self, value):
        raw_enabled = value == 2

        # Turn ON or OFF some features depending on if the user wants to use the raw BBQ or not
        self.useNAFFCheckBox.setEnabled(True == raw_enabled)
        self.secondsStep.setEnabled(True == raw_enabled)
        self.kernelSize.setEnabled(True == raw_enabled)
        self.q1Quartile.setEnabled(False == raw_enabled)
        self.q3Quartile.setEnabled(False == raw_enabled)

        # Enable or disable functionalities depending on the selected method
        if raw_enabled:
            self.useNAFFCheckBoxClicked(int(self.useNAFFCheckBox.isChecked()) * 2)  # Send a 0 or 2 depending on the state

    def useNAFFCheckBoxClicked(self, value):
        """
        Function called by the GUI when clicking the "Use NAFF" checkbox.
        This disables some functionalities only used for the raw bbq spectrogram
        """
        # Turn ON or OFF some raw bbq functions
        naff_enabled = value == 2
        self.kernelSize.setEnabled(False == naff_enabled)

    def timberVariableSelectionChanged(self, item):
        """
        Function to be called when an element of the timber selection has been changed.
        """
        if item.checkState() == Qt.Unchecked:
            if item.text() in self.selectedTimberVariables:
                self.selectedTimberVariables.remove(item.text())
            else:
                logger.error(f"Could not remove the selected timber variable '{item.text()}'")
        elif item.checkState() == Qt.Checked:
            self.selectedTimberVariables.append(item.text())

        # Update the plot accordingly
        self.updateTimberPlot(self.measurement)

    def updateTimberTable(self, measurement):
        # Get the available variables from the extracted data
        available_variables = get_variables_names_from_csv(measurement.path / timber.constants.FILENAME)
        for variable in available_variables:
            item = QListWidgetItem(variable)
            item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            item.setCheckState(Qt.Unchecked)
            self.timberVariablesListWidget.addItem(item)

    def updateTimberPlot(self, measurement):
        # Remove the existing plot, if any
        if self.plotTimberWidget is not None:
            self.deletePlot(self.plotTimberWidget)

        # Create the Matplotlib widgets
        self.plotTimberWidget = MplWidget()

        # Add the widgets to the layout
        self.timberDataLayout.addWidget(self.plotTimberWidget)

        plot_timber(self.plotTimberWidget.canvas.fig,
                    self.plotTimberWidget.canvas.ax,
                    measurement.path / timber.constants.FILENAME,
                    self.selectedTimberVariables)

        self.plotTimberWidget.canvas.draw()
        self.plotTimberWidget.show()

    def setChromaticityOrders(self, orders):
        for order in range(3, max(orders) + 1):
            dq = getattr(self, f'ChromaOrder{order}CheckBox')
            if order in orders:
                dq.setChecked(True)
            else:
                dq.setChecked(False)

    def openMeasurementCorrectionB1Clicked(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select the Optics directory of the measurement to correct",
            str(self.config.measurements_path),
        )
        if folder:
            self.measurementCorrectionB1LineEdit.setText(folder)

    def openMeasurementCorrectionB2Clicked(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select the Optics directory of the measurement to correct",
            str(self.config.measurements_path),
        )
        if folder:
            self.measurementCorrectionB2LineEdit.setText(folder)

    def correctionButtonClicked(self):
        optics_path_B1 = self.measurementCorrectionB1LineEdit.text()
        optics_path_B2 = self.measurementCorrectionB2LineEdit.text()

        observables = self.observablesCorrectionComboBox.currentData()
        chroma_factor = self.factorChromaSpinBox.value()
        rcond = self.rcondCorrectionSpinBox.value()
        method = self.correctionMethodComboBox.currentText()
        keep_dq3_constant = self.keepDQ3ConstantcheckBox.isChecked()
        clean_nan = self.cleanNaNCheckBox.isChecked()
        clean_outliers = self.cleanOutliersCheckBox.isChecked()
        clean_IR = self.cleanIRSpinBox.value()

        if len(observables) == 0:
            logger.error("No observables selected!")
            return

        optics_paths = {}
        if optics_path_B1.strip() != "":
            optics_paths[1] = optics_path_B1
        if optics_path_B2.strip() != "":
            optics_paths[2] = optics_path_B2

        # Check if we need the optics analysis or not
        rdt_in_observables = any(["f" in obs for obs in observables])
        if len(optics_paths) == 0 and rdt_in_observables:
            logger.error("No measurement path selected!")
            return

        logger.info("Starting Response Matrix creation")
        self.startThread("computeCorrections",
                         "correctionsFinished",
                         optics_paths,
                         self.measurement.path,
                         method,
                         observables,
                         chroma_factor,
                         rcond,
                         keep_dq3_constant,
                         clean_nan,
                         clean_outliers,
                         clean_IR)

    def correctionsFinished(self):
        for beam in self.corrections.keys():
            if self.corrections[beam] is None:
                continue

            text = self.corrections[f'{beam}'].replace('\n', '<br>')
            text = text.replace(" ", "&nbsp;")
            text_edit = getattr(self, f'correction{beam}TextEdit')
            text_edit.setHtml(text)
        logger.info("Corrections done!")

    # === Matplotlib rcParams
    def rcParamsClicked(self):
        rcparams_dialog = MplRcParamsDialog(self)
        rcparams_dialog.show()

class MplRcParamsDialog(QDialog, rcparams_window_class):
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setupUi(self)
        self.openRcParams()

    def openRcParams(self):
        """
        Opens the user rcParams or the default one for the GUI
        """
        if findMainWindow().config.rcParams is not None:
            text = findMainWindow().config.rcParams
        else:
            original_rcparams = RESOURCES / "chroma_gui.mplstyle"
            text = ""
            with open(original_rcparams) as f:
                text = f.read()

        self.rcParamsTextEdit.setText(text)

    def accept(self, buttonClicked):
        """
        Saves the rcParams to the resource folder.
        Can be triggered by clicking "Save" and "Apply".
        "Save" closes the window after having saved the file.
        "Apply" can be used to keep it open while tinkering with graphs
        """
        rcParams = self.rcParamsTextEdit.toPlainText()
        if rcParams.strip() == "":
            logger.warning("The supplied rcParams are empty! The file will not be overwritten.")
            return

        # Set the field and then save it
        findMainWindow().config.rcParams = rcParams
        findMainWindow().config.save_field("rcParams", rcParams)

        # Apply the new style
        findMainWindow().applyMplStyle()

        clicked_role = self.buttonBox.buttonRole(buttonClicked)
        if clicked_role == self.buttonBox.AcceptRole:  # Close the dialog
            self.close()

        logger.info("Matplotlib rcParams written")

class NewMeasurementDialog(QDialog, new_measurement_class):
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setupUi(self)

    def openLocationClicked(self):
        main_window = findMainWindow()
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select new Measurement Directory",
            str(main_window.config.measurements_path),
        )
        if folder:
            self.locationLineEdit.setText(folder)

    def openModelB1Clicked(self):
        folder = self.openModel()
        if folder:
            self.modelB1LineEdit.setText(folder)

    def openModelB2Clicked(self):
        folder = self.openModel()
        if folder:
            self.modelB2LineEdit.setText(folder)

    def openModel(self):
        main_window = findMainWindow()
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Model Directory",
            str(main_window.config.model_path),
        )
        return folder

    def createMeasurement(self):
        model_path = {'B1': Path(self.modelB1LineEdit.text()),
                      'B2': Path(self.modelB2LineEdit.text())
                      }
        # Set the start and end time of the timber extraction to the current date
        now = datetime.now()

        measurement = Measurement(path=Path(self.locationLineEdit.text()),
                                  description=self.descriptionTextEdit.toPlainText(),
                                  model_path=model_path,
                                  start_time=now,
                                  end_time=now)
        measurement.save_as_json()

        main_window = findMainWindow()
        main_window.measurement = measurement
        main_window.updateLineEdits()
        self.close()


def findMainWindow() -> typing.Union[QMainWindow, None]:
    # Global function to find the (open) QMainWindow in application
    app = QApplication.instance()
    for widget in app.topLevelWidgets():
        if isinstance(widget, QMainWindow):
            return widget
    return None


def main(argv):
    logger.info(f"Running the chroma-gui with python: {sys.executable}")
    # Setup an exception catcher so the app does not crash
    sys.excepthook = exceptHook

    # Start the app
    app = QApplication(argv)
    ChromaGui = MainWindow(None)
    ChromaGui.show()
    app.exec()


if __name__ == "__main__":
    main(sys.argv)
