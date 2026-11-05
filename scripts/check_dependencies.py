"""Check dependency metadata, allowing only decord 0.6.0's known bad wheel tag.

The upstream py3-none wheel contains a cp36-cp36m tag internally. Its ctypes
library is exercised by check_environment.py; do not suppress any other error.
"""
import subprocess
import sys

result = subprocess.run([sys.executable, "-m", "pip", "check"],
                        capture_output=True, text=True)
print(result.stdout, end="")
print(result.stderr, end="", file=sys.stderr)
if result.returncode:
    lines = (result.stdout + result.stderr).strip().splitlines()
    if lines == ["decord 0.6.0 is not supported on this platform"]:
        print("Known decord wheel metadata defect; actual video decoding is checked separately.")
    else:
        sys.exit(result.returncode)
