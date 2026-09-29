"""Regenerate the plotted quantities from the bundled compact data."""
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parent
for name in ("make_figures.py","analyze_2d_order.py","analyze_axial_order.py"):
    subprocess.run([sys.executable,str(ROOT/"prl_manuscript"/name)],check=True,cwd=ROOT)
output=ROOT/"prl_manuscript"/"figures"/"v10"
expected=[output/f"fig{i}.png" for i in range(1,5)] + [output/f"figS{i}.png" for i in range(1,5)]
missing=[str(p) for p in expected if not p.is_file()]
if missing: raise RuntimeError("Missing plots: "+", ".join(missing))
print("Eight manuscript plots generated in",output)
