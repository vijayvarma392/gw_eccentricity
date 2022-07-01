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
    default=[-1],
    nargs="+",
    help=("Generate waveforms for the ecc values.\n"
          "Default is -1 which means it will generate `args.num` waveforms"
          " from `10**args.emin` to `10**args.emax`\n"
          "usage: --ecc 0.1 0.2"))
parser.add_argument(
    "--Momega0",
    type=float,
    default=0.01,
    help="Initial Momega0 to generate waveform. Default is 0.01.")
parser.add_argument(
    "--emin",
    type=float,
    default=-7.0,
    help="Minimum eccentricity in log10.")
parser.add_argument(
    "--emax",
    type=float,
    default=0.0,
    help="Maximum eccentricity in log10.")
parser.add_argument(
    "--num",
    type=int,
    default=150,
    help="Number of waveforms")

args = parser.parse_args()
if -1 in args.ecc:
    args.ecc = 10.0**np.linspace(args.emin, args.emax,
                                 args.num)

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

mean_ano = np.pi/2
M = 50
MT = M * lal.MTSUN_SI
freq_in = args.Momega0 / (MT * np.pi)
print(freq_in)
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
            f_min=freq_in,
            eccentricity=ecc,
            eccentric_anomaly=mean_ano,
            EccIC=-1,
            physical_units=False)
        fileName = (f"{data_dir}/EccTest_q{q:.2f}_chi1z{chi1z:.2f}_"
                    f"chi2z{chi2z:.2f}_EOBecc{ecc:.10f}_"
                    f"Momega0{args.Momega0:.3f}_meanAno{mean_ano:.3f}.h5")
        f = h5py.File(fileName, "w")
        f["(2, 2)"] = modes[2, 2]
        f["t"] = times
        dset = f.create_dataset("params", ())
        dset.attrs["q"] = q
        dset.attrs["deltaTOverM"] = deltaTOverM
        dset.attrs["Momega0"] = args.Momega0
        dset.attrs["ecc"] = ecc
        dset.attrs["mean_ano"] = mean_ano
        dset.attrs["chi1z"] = chi1z
        dset.attrs["chi2z"] = chi2z
        f.close()

print(f"files are saved at {data_dir}")
