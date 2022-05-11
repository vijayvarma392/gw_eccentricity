"""Wrapper to run all tests in this directory."""
import os
import glob
import argparse
import sys
sys.path.append("../../")
from measureEccentricity.utils import SmartFormatter


__doc__ = """This script runs all the tests in the directory.
This should produce the example figures in the directory.
The tests are run for the parameter set 3 for all
available methods.
Usage: python run_all_tests.py -d ecc_waveforms"""

parser = argparse.ArgumentParser(
    description=(__doc__),
    formatter_class=SmartFormatter)
parser.add_argument(
    "--data_dir", "-d",
    type=str,
    required=True,
    help=("Base directory where waveform files are stored. You can get this "
          "from home/md.shaikh/ecc_waveforms on CIT."))
args = parser.parse_args()

tests = glob.glob("test_*.py")

print("The following tests would be performed")
for idx, test in enumerate(tests):
    print(f"{idx + 1}: {os.path.basename(test)}")

for test in tests:
    os.system(f"python {test} -d {args.data_dir} -p 3 --example")
