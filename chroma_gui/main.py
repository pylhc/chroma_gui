from datetime import datetime
import typing
import ast
import sys
import json
from json import JSONDecodeError
import tfs
from typing import Tuple, List

import timber
import logging
from pathlib import Path
from dataclasses import dataclass, field
import traceback

# PyQt libraries
from PyQt5.QtGui import QPalette, QStandardItem, QFontMetrics
from PyQt5.QtCore import QDir, QDateTime, pyqtSignal, QThread, QAbstractTableModel, QModelIndex, Qt, QEvent, QRect
from PyQt5 import uic, QtCore
from PyQt5.QtWidgets import (
    QMainWindow,
    QApplication,
    QDialog,
    QFileDialog,
    QWidget,
    QMessageBox,
    QTableView,
    QSizePolicy,
    QHeaderView,
    QComboBox, QStyledItemDelegate, qApp,
)

# Chroma-GUI specific libraries
from plotting.widget import MplWidget, mathTex_to_QPixmap
from plotting import plot_dpp, plot_freq
from cleaning import plateau, clean
from chromaticity import (
    get_chromaticity,
    construct_chroma_tfs,
    plot_chromaticity,
    get_maximum_chromaticity,
    get_chromaticity_df_with_notation,
    get_chromaticity_formula
)
import cleaning.constants
from constants import CHROMA_FILE, RESPONSE_MATRICES, CONFIG
from corrections import correct

logger = logging.getLogger('chroma_GUI')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

new_measurement_class = uic.loadUiType(Path(__file__).parent / "new_measurement.ui")[0]  # Load the UI
main_window_class = uic.loadUiType(Path(__file__).parent / "chroma_gui.ui")[0]


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


class Measurement:
    """
    Holds Measurement specific data such as paths
    """

    def __init__(self, path, description=None, model_path=None, nominal_rf=None, start_time=None, end_time=None):
        self.path = Path(path)
        self.description = description
        self.model_path = model_path
        self.nominal_rf = nominal_rf
        self.start_time: datetime = start_time
        self.end_time: datetime = end_time

        # Momentum compaction factor, retrieved from the model
        self.alpha = {"B1": None,
                      "B2": None,
                      }

        # When opening a measurement, the model path is not specified.
        # Retrieve all the info from the measurement.info file
        if not self.model_path:
            measurement_info_path = Path(self.path) / 'measurement.info'
            measurement_info = json.load(open(measurement_info_path))
            self.description = measurement_info['description']
            self.model_path = measurement_info['model_path']
            self.nominal_rf = measurement_info['nominal_rf']
            self.start_time = datetime.fromisoformat(measurement_info['start_time'])
            self.end_time = datetime.fromisoformat(measurement_info['end_time'])

        self.load_twiss()  # load the twiss files containing alfa
        self.updateLineEdits()  # Update the interface to show the data

    def load_twiss(self):
        for beam in self.alpha.keys():
            twiss = tfs.read(Path(self.model_path[beam]) / 'twiss.dat')
            self.alpha[beam] = twiss.headers['ALFA']

    def updateLineEdits(self):
        # Update the labels in the Main Window
        main_window = findMainWindow()
        main_window.alfaB1LineEdit.setText(str(self.alpha['B1']))
        main_window.alfaB2LineEdit.setText(str(self.alpha['B2']))
        main_window.nominalRfLineEdit.setText(str(self.nominal_rf))
        main_window.descriptionPlainTextEdit.setPlainText(self.description)

        # Set the extraction dates via Qt objects
        start = QDateTime.fromString(self.start_time.strftime("%Y-%m-%dT%H:%M:%S"), 'yyyy-MM-ddThh:mm:ss')
        end = QDateTime.fromString(self.end_time.strftime("%Y-%m-%dT%H:%M:%S"), 'yyyy-MM-ddThh:mm:ss')
        main_window.startTimeTimberEdit.setDateTime(start)
        main_window.endTimeTimberEdit.setDateTime(end)

        # Write if an extraction already exists with some details about the date
        message, extracted = self.get_timber_status()
        main_window.statusTimberExtractionLabel.setText(message)

        # Enable the Timber tab for the first time
        main_window.enableTimberTab(True)

        # Enable the cleaning tab if we've got extracted data
        if extracted:
            main_window.enableCleaningTab(True)
            # Set the times
            start = QDateTime.fromString(self.start_time.strftime("%Y-%m-%dT%H:%M:%S"), 'yyyy-MM-ddThh:mm:ss')
            end = QDateTime.fromString(self.end_time.strftime("%Y-%m-%dT%H:%M:%S"), 'yyyy-MM-ddThh:mm:ss')
            main_window.startPlateauDateTimeEdit.setDateTime(start)
            main_window.endPlateauDateTimeEdit.setDateTime(end)

        # Check if we got cleaned data already
        dpp, cleaned = self.get_cleaning_status()
        if dpp:  # If we have created plateaus, plot them
            main_window.plateauFinished(self)
        else:  # If we don't have plateaus, there is no need to display the cleaning yet
            main_window.cleaningWidget.setEnabled(False)

        if cleaned:  # the cleaned data exist, it can be plotted!
            main_window.cleaningFinished(self)
            main_window.enableChromaticityTab(True)

        chroma_status, chroma_orders = self.get_chroma_status()
        if chroma_status:
            main_window.setChromaticityOrders(chroma_orders)
            main_window.chromaFinished(self)

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
        path_dpp = Path(self.path / cleaning.constants.DPP_FILE.format(beam=1))
        path_cleaned = Path(self.path / cleaning.constants.CLEANED_DPP_FILE.format(beam=1))
        return path_dpp.exists(), path_cleaned.exists()

    def get_chroma_status(self):
        path_chroma = Path(self.path / CHROMA_FILE)
        if path_chroma.exists():
            chroma_tfs = tfs.read(path_chroma)
            return True, (chroma_tfs.headers['MIN_FIT_ORDER'], chroma_tfs.headers['MAX_FIT_ORDER'])
        return False, None

    def save_as_json(self):
        measurement_info_path = Path(self.path) / 'measurement.info'
        start = self.start_time.isoformat() if self.start_time else None
        end = self.end_time.isoformat() if self.end_time else None
        data = {"path": str(self.path),
                "model_path": {"B1": self.model_path['B1'],
                               "B2": self.model_path['B2']
                               },
                "description": self.description,
                "nominal_rf": self.nominal_rf,
                "start_time": start,
                "end_time": end,
                }
        json.dump(data, open(measurement_info_path, 'w'), indent=4)


class ExternalProgram(QThread):
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def __init__(self, *args):
        super(QThread, self).__init__()
        self.args = args

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
            timber.extract.save_as_pickle(measurement_path, data)

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

        # Beam 1
        clean.clean_data_for_beam(input_file_B1, output_path, output_filename_B1, qx_window, qy_window, quartiles,
                                  plateau_length, bad_tunes)

        # Beam 2
        clean.clean_data_for_beam(input_file_B2, output_path, output_filename_B2, qx_window, qy_window, quartiles,
                                  plateau_length, bad_tunes)
        self.finished.emit()

    def computeChroma(self):
        input_file_B1, input_file_B2, output_path, fit_orders, dpp_range = self.args

        # Construct the resulting TFS
        chroma_tfs = construct_chroma_tfs(fit_orders)

        # Beam 1
        chroma_tfs = get_chromaticity(input_file_B1, chroma_tfs, dpp_range, fit_orders, 'X')
        chroma_tfs = get_chromaticity(input_file_B1, chroma_tfs, dpp_range, fit_orders, 'Y')

        # Beam 2
        chroma_tfs = get_chromaticity(input_file_B2, chroma_tfs, dpp_range, fit_orders, 'X')
        chroma_tfs = get_chromaticity(input_file_B2, chroma_tfs, dpp_range, fit_orders, 'Y')

        tfs.write(output_path / CHROMA_FILE, chroma_tfs)

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
    model_path: Path = Path('.')
    measurements_path: Path = Path('.')

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

    @classmethod
    def from_dict(cls: typing.Type["Config"], obj: dict):
        return cls(
            **obj
        )


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
    return


class MainWindow(QMainWindow, main_window_class):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.setupUi(self)
        self.measurement = None
        self.config = None

        # Thread
        self.thread = None
        self.worker = None

        # Define the widgets for the plots
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

        # Disable tabs for now, as no measurement has been created or opened yet
        # TODO add Correction tab!
        self.enableTimberTab(False)
        self.enableCleaningTab(False)
        self.enableChromaticityTab(False)

        self.setCorrectionComboBox()

    def loadConfig(self):
        if not CONFIG.exists():
            return
        config_dict = json.load(open(CONFIG))
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

        # Display the available corrections
        self.observables = json.load(open(RESPONSE_MATRICES))['AVAILABLE_OBSERVABLES']
        self.observablesCorrectionComboBox.addItems(self.observables)

    def enableTimberTab(self, value):
        self.timberTab.setEnabled(value)

    def enableCleaningTab(self, value):
        self.cleaningTab.setEnabled(value)

    def enableChromaticityTab(self, value):
        self.chromaticityTab.setEnabled(value)

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
            self.measurement = Measurement(folder)
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

        # Start the plateau creation
        logger.info("Starting Plateau Creation")
        self.startThread("createPlateaus", "plateauFinished", self.measurement.path, timber.constants.FILENAME,
                         rf_beam, start, end, nominal_rf, self.measurement.alpha)

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

        # Bad tune lines: list of tuples, e.g. [(0.26, 0.28), ]
        try:
            bad_tunes = ast.literal_eval(self.badTunesLineEdit.text())
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            bad_tunes = []
            logger.error("Could not load the bad tunes line. Defaulting to none.")

        logger.info("Starting Tune Cleaning")
        self.startThread("cleanTune",
                         "cleaningFinished",
                         self.measurement.path / cleaning.constants.DPP_FILE.format(beam=1),
                         self.measurement.path / cleaning.constants.DPP_FILE.format(beam=2),
                         self.measurement.path,
                         cleaning.constants.CLEANED_DPP_FILE.format(beam=1),
                         cleaning.constants.CLEANED_DPP_FILE.format(beam=2),
                         (qx_low, qx_high),
                         (qy_low, qy_high),
                         (q1_quartile, q3_quartile),
                         plateau_length,
                         bad_tunes)

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
        self.enableChromaticityTab(True)
        if not measurement:
            measurement = self.measurement
        self.updateCleanedTunePlot(measurement)

    def timberExtractionFinished(self):
        # Update some UI
        self.measurement.updateLineEdits()

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
        file_path = measurement.path / cleaning.constants.DPP_FILE.format(beam=1)
        plot_dpp(self.plotDppB1Widget.canvas.fig, self.plotDppB1Widget.canvas.ax, file_path)
        self.plotDppB1Widget.canvas.draw()
        self.plotDppB1Widget.show()

        # Beam 2
        file_path = measurement.path / cleaning.constants.DPP_FILE.format(beam=2)
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
        filepath = measurement.path / cleaning.constants.DPP_FILE.format(beam=1)
        plot_freq(self.plotRawTuneB1Widget.canvas.fig, self.plotRawTuneB1Widget.canvas.ax, filepath,
                  'Raw Tune Measurement for Beam 1', dpp_flag=dpp_flag, delta_rf_flag=delta_rf_flag)
        self.plotRawTuneB1Widget.canvas.draw()
        self.plotRawTuneB1Widget.show()

        # Beam 2
        filepath = measurement.path / cleaning.constants.DPP_FILE.format(beam=2)
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
        filepath = measurement.path / cleaning.constants.CLEANED_DPP_FILE.format(beam=1)
        plot_freq(self.plotCleanTuneB1Widget.canvas.fig, self.plotCleanTuneB1Widget.canvas.ax, filepath,
                  'Cleaned Tune Measurement for Beam 1', dpp_flag=dpp_flag, delta_rf_flag=delta_rf_flag, plot_style="line")
        self.plotCleanTuneB1Widget.canvas.draw()
        self.plotCleanTuneB1Widget.show()

        # Beam 2
        filepath = measurement.path / cleaning.constants.CLEANED_DPP_FILE.format(beam=2)
        plot_freq(self.plotCleanTuneB2Widget.canvas.fig, self.plotCleanTuneB2Widget.canvas.ax, filepath,
                  f'Cleaned Tune Measurement for Beam 2', dpp_flag=dpp_flag, delta_rf_flag=delta_rf_flag, plot_style="line")
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

        input_file_B1 = self.measurement.path / cleaning.constants.CLEANED_DPP_FILE.format(beam=1)
        input_file_B2 = self.measurement.path / cleaning.constants.CLEANED_DPP_FILE.format(beam=2)
        output_path = self.measurement.path
        fit_orders = self.getChromaticityOrders()
        dpp_range = (dpp_range_low, dpp_range_high)

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

    def updateChromaTablesForBeam(self, current_beam):
        current_beam += 1  # index starts at 0
        self.beamChromaticityTableView.setModel(getattr(self, f"chromaB{current_beam}TableModel"))

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
        self.beamChromaticityTableView.horizontalHeader().resizeSections(QHeaderView.Stretch)

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
        dpp_file_b1 = measurement.path / cleaning.constants.CLEANED_DPP_FILE.format(beam=1)
        plot_chromaticity(self.plotChromaB1XWidget.canvas.fig, self.plotChromaB1XWidget.canvas.ax,
                          dpp_file_b1, chroma_tfs_file, 'X', fit_orders, "B1")
        plot_chromaticity(self.plotChromaB1YWidget.canvas.fig, self.plotChromaB1YWidget.canvas.ax,
                          dpp_file_b1, chroma_tfs_file, 'Y', fit_orders, "B1")

        # Beam 2
        dpp_file_b2 = measurement.path / cleaning.constants.CLEANED_DPP_FILE.format(beam=2)
        plot_chromaticity(self.plotChromaB2XWidget.canvas.fig, self.plotChromaB2XWidget.canvas.ax,
                          dpp_file_b2, chroma_tfs_file, 'X', fit_orders, "B2")
        plot_chromaticity(self.plotChromaB2YWidget.canvas.fig, self.plotChromaB2YWidget.canvas.ax,
                          dpp_file_b2, chroma_tfs_file, 'Y', fit_orders, "B2")

        self.plotChromaB1XWidget.canvas.draw()
        self.plotChromaB1XWidget.show()

    def setChromaticityOrders(self, orders):
        for order in range(3, max(orders)+1):
            dq = getattr(self, f'ChromaOrder{order}CheckBox')
            if order in orders:
                dq.setChecked(True)
            else:
                dq.setChecked(False)

    def openMeasurementCorrectionClicked(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select the Optics directory of the measurement to correct",
            str(self.config.measurements_path),
        )
        if folder:
            self.measurementCorrectionLineEdit.setText(folder)

    def correctionButtonClicked(self):
        # TODO
        logger.info("Starting Response Matrix creation")
        optics_path = self.measurementCorrectionLineEdit.text()
        observables = self.observablesCorrectionComboBox.currentData()
        chroma_factor = self.factorChromaSpinBox.value()

        if len(observables) == 0:
            logger.error("No observables selected!")
            return

        response = correct.create_response_matrix_f1004_dq3(observables, chroma_factor, beam=1)
        logger.info('Response matrix created')
        logger.info(f'  Number of observables: {response.shape[0]}')
        logger.info(f'  Number of correctors: {response.shape[1]}\n')


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
        model_path = {'B1': self.modelB1LineEdit.text(),
                      'B2': self.modelB2LineEdit.text()}
        # Set the start and end time of the timber extraction to the current date
        now = datetime.now()

        measurement = Measurement(path=self.locationLineEdit.text(),
                                  description=self.descriptionTextEdit.toPlainText(),
                                  model_path=model_path,
                                  start_time=now,
                                  end_time=now)
        measurement.save_as_json()

        main_window = findMainWindow()
        main_window.measurement = measurement

        self.close()


def findMainWindow() -> typing.Union[QMainWindow, None]:
    # Global function to find the (open) QMainWindow in application
    app = QApplication.instance()
    for widget in app.topLevelWidgets():
        if isinstance(widget, QMainWindow):
            return widget
    return None


def main(argv):
    # Setup an exception catcher so the app does not crash
    sys.excepthook = exceptHook

    # Start the app
    app = QApplication(argv)
    ChromaGui = MainWindow(None)
    ChromaGui.show()
    app.exec()


if __name__ == "__main__":
    main(sys.argv)
