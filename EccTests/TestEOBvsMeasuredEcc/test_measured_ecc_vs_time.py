__doc__ = """This test checks how the measured eccentricity looks like as
a function of time for different eccentricity definitions. The EOB waveforms
that are used for this test are generated from a fixed mass ratio, spins, and
Momega0 (the initial dimless orbital frequency), with eccentricity varying
from 1e-5 to 0.5. We try to measure the eccentricity from these waveforms
using different eccentricity definitions. we plot the measured eccentricity vs
tref_out, and color the lines by the input EOB eccentricity at Momega0.
 You should check visually whether there
are glitchy features in these plots.
Usage:
python test_measured_ecc_vs_time.py -d ecc_waveforms -m 'Amplitude' 'ResidualAmplitude' -p 'all'
python test_measured_ecc_vs_time.py -d ecc_waveforms -m 'all' -p 'all'
"""

import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import argparse
import warnings
sys.path.append("../../")
from measureEccentricity import measure_eccentricity, get_available_methods
from measureEccentricity.load_data import load_waveform
from measureEccentricity.utils import SmartFormatter

parser = argparse.ArgumentParser(
    description=(__doc__),
    formatter_class=SmartFormatter)

parser.add_argument(
    "--data_dir", "-d",
    type=str,
    required=True,
    help=("Base directory where waveform files are stored. You can get this "
          "from home/md.shaikh/ecc_waveforms on CIT."))
parser.add_argument(
    "--method", "-m",
    type=str,
    nargs="+",
    default="all",
    help=("EccDefinition method to test. Can be 'all' OR one or more of the "
          f"methods in {list(get_available_methods())}."))
parser.add_argument(
    "--emax",
    type=float,
    required=False,
    help="Maximum ecc value to test.")
parser.add_argument(
    "--emin",
    type=float,
    required=False,
    help="Minimum ecc value to test.")
parser.add_argument(
    "--param_set_key", "-p",
    type=str,
    default="all",
    nargs="+",
    help=("R|Run test for this set of parameters kept fixed.\n"
          "Possible choices are 'all' OR one or more of 1, 2, 3, 4.\n"
          "1: q=1, chi1z=chi2z=0.\n"
          "2: q=2, chi1z=chi2z=0.5\n"
          "3: q=4, chi1z=chi2z=-0.6\n"
          "4: q=6, chi1z=0.4, chi2z=-0.4.\n"))
parser.add_argument(
    "--fig_dir", "-f",
    type=str,
    default='.',
    help="Directory to save figure.")
parser.add_argument(
    "--plot_format",
    type=str,
    default="png",
    help=("Format to save the plot. "
          "Can be any format that matplotlib supports."))

args = parser.parse_args()
EOBeccs = 10**np.linspace(-5, np.log10(0.5), 100)
# do the test for eccentricity values between emin and emax
if args.emin:
    EOBeccs = EOBeccs[EOBeccs >= args.emin]
if args.emax:
    EOBeccs = EOBeccs[EOBeccs <= args.emax]

cmap = cm.get_cmap("viridis")
colors = cmap(np.linspace(0, 1, len(EOBeccs)))

# Format: [q, chi1z, chi2z]
available_param_sets = {
    "1": [1, 0, 0],
    "2": [2, 0.5, 0.5],
    "3": [4, -0.6, -0.6],
    "4": [6, 0.4, -0.4]}
data_dir = args.data_dir + "/Non-Precessing/EOB/"

# Avoid raising warnings when length of the data for interpolation
# in monotonicity check is too long
extra_kwargs = {"debug": False,
                "num_orbits_to_exclude_before_merger": 2}


def plot_waveform_ecc_vs_time(method, set_key, fig, ax):
    tmaxList = []  # to keep track of minimum time in tref_out across all eccs
    tminList = []  # to keep track of maximum time in tref_out across all eccs
    ecciniList = []  # to keep track of the measured initial eccentricities
    for idx0, ecc in enumerate(EOBeccs):
        q, chi1z, chi2z = available_param_sets[set_key]
        fileName = (f"{data_dir}/EccTest_q{q:.2f}_chi1z{chi1z:.2f}_"
                    f"chi2z{chi2z:.2f}_EOBecc{ecc:.7f}.h5")
        kwargs = {"filepath": fileName}
        if "ResidualAmplitude" in args.method:
            fileName_zero_ecc = (f"{data_dir}/EccTest_q{q:.2f}_chi1z"
                                 f"{chi1z:.2f}_"
                                 f"chi2z{chi2z:.2f}_EOBecc{0:.7f}.h5")
        kwargs.update({"filepath_zero_ecc": fileName_zero_ecc,
                       "include_zero_ecc": True})
        dataDict = load_waveform(catalog="EOB", **kwargs)
        tref_in = dataDict["t"]
        try:
            tref_out, measured_ecc, mean_ano = measure_eccentricity(
                tref_in,
                dataDict,
                method,
                extra_kwargs=extra_kwargs)
            tminList.append(tref_out[0])
            tmaxList.append(tref_out[-1])
            ecciniList.append(measured_ecc[0])
            ax.plot(tref_out, measured_ecc, c=colors[idx0], label=f"{ecc:.7f}")
        except Exception:
            warnings.warn("Exception raised. Probably too small eccentricity "
                          "to detect any extrema.")
    ax.grid()
    if len(tmaxList) >= 1:
        tmin = max(tminList)  # choose the shortest
        tmax = max(tmaxList)
        ymax = max(ecciniList)
        ymin = min(EOBeccs)
        ax.set_xlim(tmin, tmax)
        ax.set_ylim(ymin, ymax)
    ax.set_ylabel("Measured Eccentricity")
    ax.set_yscale("log")
    # add text indicating the method used
    ax.text(0.95, 0.95, f"{method}", ha="right", va="top",
            transform=ax.transAxes)
    # add colorbar
    norm = mpl.colors.LogNorm(vmin=EOBeccs.min(), vmax=EOBeccs.max())
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.05)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
    cbar.set_label("EOB eccentricity at initial time",
                   size=10)
    fig.suptitle(rf"$q={q:.3f}, \chi_{{1z}}={chi1z:.3f}, \chi_{{2z}}={chi2z:.3f}$",
                 size=10)


if "all" in args.method:
    args.method = get_available_methods()
    # method_str is used in the filename for the output figure
    method_str = "all"
else:
    method_str = "_".join(args.method)

if "all" in args.param_set_key:
    args.param_set_key = list(available_param_sets.keys())

nrows = len(args.method)
for key in args.param_set_key:
    fig_name = (
        f"{args.fig_dir}/EccTest_eccVsTime_set{key}_"
        f"{method_str}_emin_{min(EOBeccs):.7f}_emax_{max(EOBeccs):.7f}"
        f".{args.plot_format}")
    fig, axarr = plt.subplots(nrows=nrows,
                              figsize=(6,
                                       3 * nrows),
                              sharex=True)
    for idx, method in tqdm(enumerate(args.method)):
        ax = axarr if nrows == 1 else axarr[idx]
        plot_waveform_ecc_vs_time(method, key, fig, ax)
        if idx == nrows - 1:
            ax.set_xlabel("time")
    plt.subplots_adjust(bottom=0.05,
                        top=0.95,
                        right=0.9,
                        hspace=0.1)
    fig.savefig(f"{fig_name}")
