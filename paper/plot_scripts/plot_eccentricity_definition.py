"""Make the plot to illustrate the eccentricity definition."""

import matplotlib.pyplot as plt
import sys
import sxs
import numpy as np
import os
import h5py
from scipy.interpolate import InterpolatedUnivariateSpline
sys.path.append("../../")
from measureEccentricity.measureEccentricity import measure_eccentricity
from measureEccentricity.plot_settings import use_fancy_plotsettings, colorsDict
from measureEccentricity.utils import get_peak_via_quadratic_fit

# We will use one NR waveform from SXS catalog to plot the measured
# eccentricity vs time and the omega_22 showing the pericenter and apocenter
# locations
# sxs_id = "SXS:BBH:1371"
sxs_id = "040"


def get_nr_data(sxs_id, lev=3, is_private=True, sim_path="./"):
    """Get NR waveform."""
    if is_private:
        data_file = (
            f"{sim_path}/{sxs_id}/Lev{lev}/"
            "rhOverM_Asymptotic_GeometricUnits_CoM.h5")
        cwd = os.getcwd()
        if not os.path.exists(data_file):
            os.chdir(f"{sim_path}")
            os.system(
                f"git annex get {sim_path}/{sxs_id}/Lev3/"
                "rhOverM_Asymptotic_GeometricUnits_CoM.h5")
            os.chdir(cwd)
        waveform_data = h5py.File(data_file, "r")
        waveform = waveform_data["Extrapolated_N2.dir"]
        wf22 = waveform["Y_l2_m2.dat"]
        time_nr = wf22[:, 0]
        h22_nr = wf22[:, 1] + 1j * wf22[:, 2]
    else:
        waveform = sxs.load(f"SXS:BBH:{sxs_id}/Lev/rhOverM",
                            extrapolation_order=2)
        time_nr = waveform.time
        h22_nr = waveform.data[:, waveform.index(2, 2)]
    amp_nr = np.abs(h22_nr)
    phase_nr = np.unwrap(np.angle(h22_nr))
    amp_interp = InterpolatedUnivariateSpline(time_nr, amp_nr)
    phase_interp = InterpolatedUnivariateSpline(time_nr, phase_nr)
    time_interped = np.linspace(time_nr[0], time_nr[-1], 10 * len(time_nr))
    h22_interped = (amp_interp(time_interped)
                    * np.exp(1j * phase_interp(time_interped)))
    return time_interped, h22_interped


t, h22 = get_nr_data(
    sxs_id,
    sim_path="/home1/md.shaikh/SimAnnex/Private/Ecc1dSur_rerun/")
t = t - get_peak_via_quadratic_fit(t, np.abs(h22))[0]
tstart = -8000
dataDict = {"t": t[t >= tstart],
            "hlm": {(2, 2): h22[t >= tstart]}}
print(dataDict)

tref_in = np.arange(-8000.0, 0.0, 0.1)
tref, ecc, mean_ano, eccMethod = measure_eccentricity(
    tref_in=tref_in,
    dataDict=dataDict,
    method="Amplitude",
    return_ecc_method=True)

use_fancy_plotsettings()
fig, ax = plt.subplots(nrows=2, figsize=(6, 4), sharex=True)
eccMethod.plot_measured_ecc(fig, ax[0])
eccMethod.plot_extrema_in_omega22(fig, ax[1])
ax[1].set_xlabel(r"$t(M)$")
ax[1].set_ylabel(r"$M\omega_{22}(t)$")
ax[0].set_ylabel(r"Eccentricity $e(t)$")
ax[0].set_xlabel("")
ax[1].set_ylim(0.008, 0.1)
ax[1].set_xlim(left=tref_in[0], right=0)

tref_mark = -5500
tref_mark, ecc_mark, mean_ano_mark, eccMethod_mark = measure_eccentricity(
    tref_in=tref_mark,
    dataDict=dataDict,
    method="Amplitude",
    return_ecc_method=True)

ax[0].axvline(tref_mark, c=colorsDict["vline"], ls="--")
ax[1].axvline(tref_mark, c=colorsDict["vline"], ls="--")
ax[0].plot(tref_mark, ecc_mark, c=colorsDict["vline"], marker=".")
plt.subplots_adjust(hspace=0.1)
fig.tight_layout()
fig.savefig("../figs/ecc_definition.pdf")

# plot the mean anomaly
fig, ax = plt.subplots(nrows=2, figsize=(6, 4), sharex=True)
ax[0].set_ylabel(r"Mean Anomaly $l(t)$")
ax[0].grid()
ax[0].axvline(tref_mark, c=colorsDict["vline"], ls="--")
ax[0].plot(tref_mark, mean_ano_mark, c=colorsDict["vline"], marker=".")
ax[0].set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax[0].set_yticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])

end = np.argmin(np.abs(eccMethod.t))
ax[1].plot(eccMethod.t[: end], eccMethod.omega22[: end],
           c=colorsDict["default"])
ax[1].set_xlabel(r"$t(M)$")
ax[1].set_ylabel(r"$M\omega_{22}(t)$")
ax[1].axvline(tref_mark, c=colorsDict["vline"], ls="--")

for idx in np.arange(len(eccMethod.peaks_location) - 1):
    t_peak = eccMethod.t[eccMethod.peaks_location[idx]]
    t_next_peak = eccMethod.t[eccMethod.peaks_location[idx + 1]]
    t = np.arange(max(t_peak, tref[0]), min(t_next_peak, tref[-1]), 0.1)
    mean_ano = 2 * np.pi * (t - t_peak) / (t_next_peak - t_peak)
    ax[0].plot(t, mean_ano, c=colorsDict["default"])
    if t_peak >= tref[0] and t_peak <= tref[-1]:
        ax[0].axvline(t_peak, c=colorsDict["peaksvline"], ls=":")
        ax[1].axvline(t_peak, c=colorsDict["peaksvline"], ls=":")
    if abs(t_next_peak - tref[-1]) < 10 * (eccMethod.t[1] - eccMethod.t[0]):
        ax[1].axvline(t_next_peak, c=colorsDict["peaksvline"], ls=":")
        ax[0].axvline(t_next_peak, c=colorsDict["peaksvline"], ls=":")
ax[1].set_ylim(0.008, 0.1)
ax[1].set_xlim(left=tref_in[0], right=0)
ax[1].grid()

plt.subplots_adjust(hspace=0.1)
fig.tight_layout()
fig.savefig("../figs/mean_ano_definition.pdf")
