import sys
from pathlib import Path

# Add src/module_3 to the path so tests can import utils and train
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "module_3"))