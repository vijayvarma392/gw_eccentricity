"""Make the plot to illustrate the eccentricity definition."""

import matplotlib.pyplot as plt
import sys
sys.path.append("../../")
from measureEccentricity.measureEccentricity import measure_eccentricity
from measureEccentricity.load_data import load_waveform
from measureEccentricity.plot_settings import use_fancy_plotsettings

# We will use one EOB waveform to plot the measured eccentricity vs time
# and the omega_22 showing the perisatron and apastron locations

waveform_kwargs = {"filepath": "/home1/md.shaikh/Eccentricity/data/ecc_waveforms/Non-Precessing/EOB/EccTest_q1.00_chi1z0.00_chi2z0.00_EOBecc0.1963232895_Momega00.010.h5",
                   "filepath_zero_ecc": "/home1/md.shaikh/Eccentricity/data/ecc_waveforms/Non-Precessing/EOB/EccTest_q1.00_chi1z0.00_chi2z0.00_EOBecc0.0000000000_Momega00.002.h5",
                   "include_zero_ecc": True}
dataDict = load_waveform("EOB", **waveform_kwargs)
tref_in = dataDict["t"]
tref, ecc, mean_ano, eccMethod = measure_eccentricity(tref_in, dataDict,
                                                      method="Amplitude",
                                                      return_ecc_method=True)

use_fancy_plotsettings()
fig, ax = plt.subplots(nrows=2, figsize=(6, 6), sharex=True)
eccMethod.plot_measured_ecc(fig, ax[0])
eccMethod.plot_extrema_in_omega(fig, ax[1])
ax[0].set_xlabel("")
ax[1].set_ylim(0.008, 0.06)
ax[1].set_xlim(right=0)
plt.subplots_adjust(hspace=0.1)
fig.tight_layout()
fig.savefig("../figs/ecc_definition.pdf")
