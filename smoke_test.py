"""Quick numerical/implementation check; not a large-system DHU test."""
from pathlib import Path
import json
import shutil
import subprocess
import tempfile
ROOT=Path(__file__).resolve().parent
compiler=shutil.which("g++")
if compiler is None: raise SystemExit("g++ (C++17) is required for this smoke test")
source=ROOT/"axial_dhu_sim"/"extended_scan"/"engine.cpp"
with tempfile.TemporaryDirectory(prefix="dhu-smoke-") as temporary:
    folder=Path(temporary);engine=folder/"engine.exe"
    subprocess.run([compiler,"-O2","-std=c++17",str(source),"-o",str(engine)],check=True)
    subprocess.run([str(engine),"--selftest","1"],check=True)
    prefix=folder/"tiny_open_run"
    args=[str(engine),"--L","64","--R","16","--W","8","--seed","7","--eps","20","--kappa","0.02","--burn","1","--duration","5","--out",str(prefix)]
    subprocess.run(args,check=True)
    meta=json.loads(Path(str(prefix)+"_meta.json").read_text())
    assert meta["frames"]==1 and meta["columns"]>426
    assert meta["max_control_balance_error"]==0 and meta["total_balance_error"]==0
print("PASS: force self-test, short open run, and particle-balance checks")
