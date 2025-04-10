""" 
Constants
---------

This module contains all the constants used in the chroma_gui.
"""
from pathlib import Path

# Config file
CONFIG: Path = Path.home() / ".chroma_gui"

# Resources
RESOURCES: Path = Path(__file__).parent / "resources"

# Chromaticity
CHROMA_FILE: str = "chromaticity.tfs"
CHROMA_COEFFS: Path = RESOURCES / "chromaticity_coefficients.json"
RESPONSE_MATRICES: Path = RESOURCES / "response_matrices.json"