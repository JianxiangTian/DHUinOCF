"""Verify the release manifest using only the Python standard library."""
from pathlib import Path
import hashlib
ROOT=Path(__file__).resolve().parent
for line in (ROOT/"MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
    digest,path=line.split("  ",1);file=ROOT/path
    assert file.is_file(),f"Missing: {path}"
    actual=hashlib.sha256(file.read_bytes()).hexdigest()
    assert actual==digest,f"Checksum mismatch: {path}"
print("PASS: all packaged file checksums match")
