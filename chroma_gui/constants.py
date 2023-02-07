from pathlib import Path

# Config file
CONFIG = Path.home() / ".chroma_gui"

# Resources
RESOURCES = Path(__file__).parent / "resources"

# Chromaticity
CHROMA_FILE = "chromaticity.tfs"
CHROMA_COEFFS = RESOURCES / "chromaticity_coefficients.json"
RESPONSE_MATRICES = RESOURCES / "response_matrices.json"