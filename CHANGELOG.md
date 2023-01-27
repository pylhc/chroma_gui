# Non-Linear Chromaticity GUI

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
