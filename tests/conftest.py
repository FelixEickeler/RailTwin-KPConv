import sys
from pathlib import Path

# Make the repository root importable (datasets, utils, kernels, ...)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
