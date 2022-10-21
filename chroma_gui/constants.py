from pathlib import Path

# Config file
CONFIG = Path.home() / ".chroma_gui"

# Chromaticity
CHROMA_FILE = "chromaticity.tfs"
RESPONSE_MATRICES = Path(__file__).parent / "resources" / "response_matrices.json"
RESOURCES = Path(__file__).parent / "resources"