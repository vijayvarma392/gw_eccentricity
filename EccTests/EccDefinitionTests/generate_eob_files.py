import os
home = os.path.expanduser("~")
import sys
sys.path.append(f"{home}/measuring_eccentricity_from_higher_modes/seobnrv4e/")

import seobnrv4ehm as seob
import numpy as np
from tqdm import tqdm
import lal
import h5py

q = 1
chi1z = 0
chi2z = 0

ecc_inis = 10**np.linspace(-5, np.log10(0.5), 100)
freq_in = 10
M = 50
MT = M * lal.MTSUN_SI
Momega0 = MT * np.pi * freq_in
deltaTOverM = 1
dt = deltaTOverM * MT
print(1/dt)

data_dir = "/home1/md.shaikh/Eccentricity/EccTests/EccDefinitionTests/EOB"

if not os.path.exists(data_dir):
    os.system(f"mkdir -p {data_dir}")

for ecc in tqdm(ecc_inis):
    times, modes = seob.get_modes(
        q=q,
        chi1=chi1z,
        chi2=chi2z,
        M_fed=M,
        delta_t=dt,
        f_min=freq_in,
        eccentricity=ecc,
        physical_units=False)
    fileName = f"{data_dir}/EccTest_q{q:.2f}_chi1z{chi1z:.2f}_chi2z{chi2z:.2f}_EOBecc{ecc:.7f}.h5"
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
