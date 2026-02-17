from pathlib import Path
import importlib

package_dir = Path(__file__).parent

for module_file in package_dir.glob("*.py"):
    if module_file.name not in ("__init__.py","base.py"):
        importlib.import_module(f"{__name__}.{module_file.stem}")