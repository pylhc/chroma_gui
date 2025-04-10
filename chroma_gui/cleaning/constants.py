"""
Constants
---------

Constants for the cleaning module.
"""

RF_VARIABLE: str = "ALB.SR4.B{beam}:FGC_FREQ"

TUNE_VARS: list[str] = [
    "LHC.BQBBQ.CONTINUOUS_HS.B{beam}:EIGEN_FREQ_1",
    "LHC.BQBBQ.CONTINUOUS_HS.B{beam}:EIGEN_FREQ_2",
]
X_VAR_INDICATOR: str = "EIGEN_FREQ_1"

DPP_FILE: str = "dpp_B{beam}.tfs"
CLEANED_DPP_FILE: str = "dpp_cleaned_B{beam}.tfs"
