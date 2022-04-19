"""
Generate EOB waveforms using Toni's SEOBNRv4EHM model.

This script is mostly for me (Arif) to generate EOB waveforms for the
test_eob_vs_measured_ecc.py script and is not important for most of the
users. Only script to run for test is test_eob_vs_measured_ecc.py.
"""
import os
import argparse
from argparse import RawTextHelpFormatter
import numpy as np
from tqdm import tqdm
import lal
import h5py
home = os.path.expanduser("~")
import sys
sys.path.append(f"{home}/measuring_eccentricity_from_higher_modes/seobnrv4e/")
import seobnrv4ehm as seob

parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
parser.add_argument(
    "--data_dir",
    type=str,
    required=True,
    help=("Base directory where waveform files are to be stored."))
parser.add_argument(
    "--set",
    type=str,
    default="all",
    nargs="+",
    help=("Generate waveforms for the set of parameters.\n"
          "Possible choices are one or more of 1, 2, 3, 4 or all.\n"
          "1: q=1, chi1z=chi2z=0.\n"
          "2: q=2, chi1z=chi2z=0.5\n"
          "3: q=4, chi1z=chi2z=-0.6\n"
          "4: q=6, chi1z=0.4, chi2z=-0.4.\n"
          "Default is 'all'.\n"
          "usage: --set 1 2"))
parser.add_argument(
    "--ecc",
    type=float,
    default=-1,
    nargs="+",
    help=("Generate waveforms for the ecc values.\n"
          "Default is -1 which means it will generate 100 waveforms"
          " from 1e-5 to 0.5\n"
          "usage: --ecc 0.1 0.2"))
parser.add_argument(
    "--freq_in",
    type=float,
    default=10,
    help="Initial frequency to generate waveform. Default is 10 Hz.")

args = parser.parse_args()

if -1 in args.ecc:
    args.ecc = 10**np.linspace(-5, np.log10(0.5), 100)

param_sets = {"1": [1, 0, 0],
              "2": [2, 0.5, 0.5],
              "3": [4, -0.6, -0.6],
              "4": [6, 0.4, -0.4]}

if "all" in args.set:
    args.set = param_sets
else:
    sets = {}
    for key in args.set:
        sets.update({key: param_sets[key]})
    args.set = sets

M = 50
MT = M * lal.MTSUN_SI
Momega0 = MT * np.pi * args.freq_in
deltaTOverM = 1
dt = deltaTOverM * MT
print(1/dt)

data_dir = args.data_dir + "Non-Precessing/EOB/"

if not os.path.exists(data_dir):
    os.system(f"mkdir -p {data_dir}")

for key in args.set:
    q, chi1z, chi2z = param_sets[key]
    for ecc in tqdm(args.ecc):
        times, modes = seob.get_modes(
            q=q,
            chi1=chi1z,
            chi2=chi2z,
            M_fed=M,
            delta_t=dt,
            f_min=args.freq_in,
            eccentricity=ecc,
            physical_units=False)
        fileName = (f"{data_dir}/EccTest_q{q:.2f}_chi1z{chi1z:.2f}_"
                    f"chi2z{chi2z:.2f}_EOBecc{ecc:.7f}.h5")
        f = h5py.File(fileName, "w")
        f["(2, 2)"] = modes[2, 2]
        f["t"] = times
        dset = f.create_dataset("params", ())
        dset.attrs["q"] = q
        dset.attrs["deltaTOverM"] = deltaTOverM
        dset.attrs["Momega0"] = Momega0
        dset.attrs["ecc"] = ecc
        dset.attrs["chi1z"] = chi1z
        dset.attrs["chi2z"] = chi2z
        f.close()
