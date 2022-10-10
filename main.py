from datetime import datetime
import typing

from PyQt5.QtWidgets import (
    QMainWindow,
    QApplication,
    QDialog,
    QFileDialog,
    QWidget,
    QMessageBox,
)
from PyQt5 import uic
from PyQt5.QtCore import QDir, QDateTime, QObject, pyqtSignal, QThread
from pathlib import Path

import sys
import json
from json import JSONDecodeError
import tfs
import timber
import logging

from plotting.widget import MplWidget
from plotting import plot_dpp, plot_freq
from cleaning import plateau
import cleaning.constants

logger = logging.getLogger('chroma_GUI')
logger.setLevel(logging.INFO)

new_measurement_class = uic.loadUiType("new_measurement.ui")[0]  # Load the UI
main_window_class = uic.loadUiType("chroma_gui.ui")[0]


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
            main_window.cleaningFinished(self)
        else:  # If we don't have plateaus, there is no need to display the cleaning yet
            main_window.cleaningWidget.setEnabled(False)

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


class MainWindow(QMainWindow, main_window_class):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.setupUi(self)
        self.measurement = None

        # Disable tabs for now, as no measurement has been created or opened yet
        # TODO add Chromaticity and Correction tabs!
        self.enableTimberTab(False)
        self.enableCleaningTab(False)

    def enableTimberTab(self, value):
        self.timberTab.setEnabled(value)

    def enableCleaningTab(self, value):
        self.cleaningTab.setEnabled(value)

    def newMeasurementClicked(self):
        measurement_dialog = NewMeasurementDialog(self)
        measurement_dialog.show()

    def openMeasurementClicked(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select new Measurement Directory",
            QDir.currentPath(),
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
        except JSONDecodeError:
            QMessageBox.warning(self,
                                "Failed to open measurement",
                                f"The file 'measurement.info' is not a valid JSON file")
        except KeyError:
            QMessageBox.warning(self,
                                "Failed to open measurement",
                                f"The file 'measurement.info' does not contain the required keys")

    def saveSettingsClicked(self):
        """
        Save the settings into the measurement file
        """
        nominal_rf = self.nominalRfLineEdit.text()
        if nominal_rf == 0 or nominal_rf == 'None':
            nominal_rf = None
        if nominal_rf is not None:
            nominal_rf = float(nominal_rf)

        self.measurement.nominal_rf = nominal_rf

        # Save the measurement
        self.measurement.save_as_json()

    def extractTimberClicked(self):
        start = self.startTimeTimberEdit.dateTime().toPyDateTime()
        end = self.endTimeTimberEdit.dateTime().toPyDateTime()

        # Set the start and end times to the Measurement object, and save it
        self.measurement.start_time = start
        self.measurement.end_time = end
        self.measurement.save_as_json()

        # Step 2: Create a QThread object
        self.thread = QThread()
        # Step 3: Create a worker object
        self.worker = ExternalProgram(start, end)
        # Step 4: Move worker to the thread
        self.worker.moveToThread(self.thread)
        # Step 5: Connect signals and slots
        self.thread.started.connect(self.worker.extractTimber)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.timberExtractionFinished)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        # Step 6: Start the thread
        self.thread.start()

    def createPlateausClicked(self):
        start = self.startPlateauDateTimeEdit.dateTime().toPyDateTime()
        end = self.endPlateauDateTimeEdit.dateTime().toPyDateTime()
        rf_beam = int(self.rfBeamComboBox.currentText())
        nominal_rf = self.nominalRfLineEdit.text()

        # Check of the nominal RF exists
        # If it is 0 or None, the plateau creation will take the first value in the dataFrame
        if nominal_rf == 0 or nominal_rf == 'None':
            nominal_rf = None
        if nominal_rf is not None:
            nominal_rf = float(nominal_rf)

        alpha = self.measurement.alpha

        logger.info("Starting Plateau Creation")
        self.thread = QThread()
        self.worker = ExternalProgram(self.measurement.path, timber.constants.FILENAME, rf_beam, start, end,
                                      nominal_rf, alpha)
        self.worker.moveToThread(self.thread)

        # Assign functions to different states of the thread
        self.thread.started.connect(self.worker.createPlateaus)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.cleaningFinished)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def cleaningFinished(self, measurement=None):
        logger.info("Cleaning done!")
        # This function is called during the Measurement object init when loading a directory
        # This means the 'measurement' object isn't assigned yet to the main window, i.e. 'self.measurement' is None
        if not measurement:
            measurement = self.measurement
        self.updateDppPlot(measurement)
        self.updateRawTunePlot(measurement)
        self.cleaningWidget.setEnabled(True)

    def timberExtractionFinished(self):
        # Update some UI
        self.measurement.updateLineEdits()

    def updateDppPlot(self, measurement):
        # Add a plot to show the DPP / RF measurement
        plotWidgetB1 = MplWidget()
        plotWidgetB2 = MplWidget()
        self.cleaningPlotB1Layout.addWidget(plotWidgetB1)
        self.cleaningPlotB2Layout.addWidget(plotWidgetB2)

        # Beam 1
        file_path = measurement.path / cleaning.constants.DPP_FILE.format(beam=1)
        plot_dpp(plotWidgetB1.canvas.fig, plotWidgetB1.canvas.ax, file_path)
        plotWidgetB1.canvas.draw()
        plotWidgetB1.show()

        # Beam 2
        file_path = measurement.path / cleaning.constants.DPP_FILE.format(beam=2)
        plot_dpp(plotWidgetB2.canvas.fig, plotWidgetB2.canvas.ax, file_path)
        plotWidgetB2.canvas.draw()
        plotWidgetB2.show()

    def updateRawTunePlot(self, measurement):
        # Add a plot to show the tune before cleaning
        plotWidgetB1 = MplWidget()
        plotWidgetB2 = MplWidget()
        self.rawPlotB1Layout.addWidget(plotWidgetB1)
        self.rawPlotB2Layout.addWidget(plotWidgetB2)

        # Beam 1
        filepath = measurement.path / cleaning.constants.DPP_FILE.format(beam=1)
        plot_freq(plotWidgetB1.canvas.fig, plotWidgetB1.canvas.ax, filepath, f'Raw Tune Measurement for Beam 1',
                  dpp_flag=False, delta_rf_flag=False)
        plotWidgetB1.canvas.draw()
        plotWidgetB1.show()

        # Beam 2
        filepath = measurement.path / cleaning.constants.DPP_FILE.format(beam=2)
        plot_freq(plotWidgetB2.canvas.fig, plotWidgetB2.canvas.ax, filepath, f'Raw Tune Measurement for Beam 2',
                  dpp_flag=False, delta_rf_flag=False)
        plotWidgetB2.canvas.draw()
        plotWidgetB2.show()


class NewMeasurementDialog(QDialog, new_measurement_class):
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setupUi(self)

    def openLocationClicked(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select new Measurement Directory",
            QDir.currentPath(),
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
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Model Directory",
            QDir.currentPath(),
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ChromaGui = MainWindow(None)
    ChromaGui.show()
    app.exec_()
