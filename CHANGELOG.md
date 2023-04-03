# Non-Linear Chromaticity GUI

# 2023-03-22 - v0.0.18

* Changed:
  * Renamed twiss.dat to twiss.tfs in corrections

# 2023-03-22 - v0.0.18

* Added:
  * Version in main window title

# 2023-03-22 - v0.0.16 - v0.0.17

Release patch

# 2023-03-21 - v0.0.15

* Added:
  * Corrections are now relative and can be used as MAD-X input

* Fixed:
  * alfa isn't written anymore in the measurement data, but taken from the
    model when needed

# 2023-02-10 - v0.0.14

* Added:
  * Global and local corrections

* Changed
  * Plateaus with large tune std are now removed

# 2023-01-31 - v0.0.13

* Add pytables are requirement, need for hdf

# 2023-01-31 - v0.0.10 - v0.0.11 - v0.0.12

* Fix unused dependencies:
  * Removed PyNAFF
  * Removed sklearn

* Set the minimum version for PyQt5

# 2023-01-30 - v0.0.9

* Changed
  * Removed setup.py in favor of pyproject.toml and setup.cfg
    The Gitlab-CI has been updated as well.

# 2023-01-30 - v0.0.8

* Removed:
  * Harpy support, too slow to be usable and not meant for that usage

* Added:
  * Bad lines usage for raw BBQ processing
  * Progress bar for cleaning

# 2023-01-27 - v0.0.7

* Changed:
  * Using HDF instead of pickle to stay compatible with different pandas
    versions

# 2023-01-27 - v0.0.6

Many changes for this release, mainly about the raw BBQ processing.

Added:
  * Raw BBQ processing based on several methods 
    * spectrogram and median filter
    * OMC3 harpy
    * pyNAFF
  * Window to change matplotlib's rcParams
  * Reduced chi square in the chromaticity tab
  * Info icons are added automatically to QLabels with a tooltip

## 2022-11-06 - v0.0.5

Fixed:
  * Config would not be loaded properly when the file did not exist

## 2022-11-06 - v0.0.4

Added:
  * Button to copy the chromaticity table to the clipboard
  * Button save plots in the chromaticity tab

Fixed:
  * Line edits and plots would not update when creating a measurement
