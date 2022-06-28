__doc__ = """This test checks whether different eccentricity definitions vary
smoothly as a function of the internal eccentricity definition used by Toni's
EOB model. The EOB waveforms that are used for this test are generated from a
fixed mass ratio, spins, and Momega0 (the initial dimless orbital frequency),
with eccentricity varying from 1e-7 to 1. We try to measure the eccentricity
from these waveforms using different eccentricity definitions. For each
waveform, we measure the eccentricity at the very first extrema (periastron or
apastron). That way, the measured eccentricity is also at (nearly) Momega0.
Finally, we plot the measured eccentricity vs the internal EOB eccentricity.
You should check visually whether there are glitchy features in these plots.
Usage:
python test_eob_vs_measured_ecc.py -d ecc_waveforms -m 'Amplitude' 'ResidualAmplitude' -p 'all'
python test_eob_vs_measured_ecc.py -d ecc_waveforms -m 'all' -p 'all'
"""

import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
import warnings
sys.path.append("../../")
from gw_eccentricity import measure_eccentricity, get_available_methods
from gw_eccentricity.load_data import load_waveform
from gw_eccentricity.utils import SmartFormatter
from gw_eccentricity.plot_settings import use_fancy_plotsettings, figWidthsOneColDict
from gw_eccentricity.plot_settings import colorsDict, lwidths, lalphas, lstyles

parser = argparse.ArgumentParser(
    description=(__doc__),
    formatter_class=SmartFormatter)

parser.add_argument(
    "--data_dir", "-d",
    type=str,
    default="../../data/ecc_waveforms",
    help=("Base directory where waveform files are stored. You can get this "
          "from home/md.shaikh/ecc_waveforms on CIT."))
parser.add_argument(
    "--method", "-m",
    type=str,
    nargs="+",
    default="all",
    help=("EccDefinition method to test. Can be 'all' OR one or more of the "
          f"methods in {get_available_methods()}."))
parser.add_argument(
    "--param_set_key", "-p",
    type=str,
    default="all",
    nargs="+",
    help=("Run test for this set of parameters kept fixed.\n"
          "Possible choices are 'all' OR one or more of 1, 2, 3, 4.\n"
          "1: q=1, chi1z=chi2z=0.\n"
          "2: q=2, chi1z=chi2z=0.5\n"
          "3: q=4, chi1z=chi2z=-0.6\n"
          "4: q=6, chi1z=0.4, chi2z=-0.4.\n"))
parser.add_argument(
    "--fig_dir",
    "-f",
    type=str,
    default='.',
    help="Directory to save figure.")
parser.add_argument(
    "--plot_format",
    type=str,
    default="png",
    help=("Format to save the plot. "
          "Can be any format that matplotlib supports."))
parser.add_argument(
    "--example",
    action="store_true",
    help=("This will override the figure name (that contains the "
          "information about parameter set, method used and so on)"
          " and uses a figure name which is of the form test_name_example.png"
          "where test_name is the name of the test."))
parser.add_argument(
    "--slice",
    type=int,
    default=1,
    help=("Slice the EOBeccs array by taking only every nth ecc value given by"
          " slice. This is useful when we want do not want to loop over all"
          "the ecc but skip n number of ecc given by slice"))
parser.add_argument(
    "--paper",
    action="store_true",
    help="Remove markers for paper.")

args = parser.parse_args()

EOBeccs = 10**np.linspace(-7, 0, 150)
Momega0 = 0.01
meanAno = np.pi/2
Momega0_zeroecc = 0.002
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
                # "treat_mid_points_between_peaks_as_troughs": True
                }

cmap = cm.get_cmap("plasma")
colors = cmap(np.linspace(0, 1, len(EOBeccs)))


def plot_waveform_ecc_vs_model_ecc(methods, key):
    # Get the output figure name
    # method_str is used in the filename for the output figure
    if "all" in methods:
        methods = get_available_methods()
        method_str = "all"
    else:
        method_str = "_".join(args.method)

    if args.example:
        fig_eob_vs_measured_ecc_name = (f"{args.fig_dir}/test_eob_vs_measured_ecc_example"
                                        f".{args.plot_format}")
        fig_measured_ecc_vs_time_name = (f"{args.fig_dir}/test_measured_ecc_vs_time_example."
                    f"{args.plot_format}")
    else:
        fig_eob_vs_measured_ecc_name = (f"{args.fig_dir}/test_eob_vs_measured_ecc_set{key}_{method_str}"
                                   f".{args.plot_format}")
        fig_measured_ecc_vs_time_name = (f"{args.fig_dir}/test_measured_ecc_vs_time_set{key}_{method_str}"
                                         f".{args.plot_format}")
    # For eob vs measured ecc
    fig_eob_vs_measured_ecc, ax_eob_vs_measured_ecc = plt.subplots(
        figsize=(figWidthsOneColDict[journal], 3))
    # For measured ecc vs time
    fig_measured_ecc_vs_time, ax_measured_ecc_vs_time = plt.subplots(
        nrows=len(methods),
        figsize=(figWidthsOneColDict[journal], # figure width 
                 (2 if args.paper else 3) # height of each row
                 * len(methods) # number of rows
                 ))
    # create dictionary to store different quantities for each methods
    # as we loop over all the eccentricity and methods
    model_eccs = {} # to keep track of eob input eccentricity
    measured_eccs = {}  # to keep track of ecc as measured by the definition
    tmaxList = {}  # to keep track of minimum time in tref_out across all eccs
    tminList = {}  # to keep track of maximum time in tref_out across all eccs
    # Initiate the dictionary with an empty list for each method
    for method in methods:
        model_eccs.update({method: []})
        measured_eccs.update({method: []})
        tmaxList.update({method: []})
        tminList.update({method: []})
    q, chi1z, chi2z = available_param_sets[key]
    for idx0, ecc in tqdm(enumerate(EOBeccs)):
        fileName = (f"{data_dir}/EccTest_q{q:.2f}_chi1z{chi1z:.2f}_"
                    f"chi2z{chi2z:.2f}_EOBecc{ecc:.10f}_"
                    f"Momega0{Momega0:.3f}_meanAno{meanAno:.3f}.h5")
        kwargs = {"filepath": fileName}
        for idx, method in enumerate(methods):
            if "Residual" in method:
                fileName_zero_ecc = (
                    f"{data_dir}/EccTest_q{q:.2f}_chi1z"
                    f"{chi1z:.2f}_"
                    f"chi2z{chi2z:.2f}_EOBecc{0:.10f}_"
                    f"Momega0{Momega0_zeroecc:.3f}_meanAno{meanAno:.3f}.h5")
                kwargs.update({"filepath_zero_ecc": fileName_zero_ecc,
                               "include_zero_ecc": True})
            dataDict = load_waveform(catalog="EOB", **kwargs)
            tref_in = dataDict["t"]
            try:
                tref_out, measured_ecc, mean_ano = measure_eccentricity(
                    tref_in=tref_in,
                    dataDict=dataDict,
                    method=method,
                    extra_kwargs=extra_kwargs)
                model_eccs[method] += [ecc]
                # Get the measured eccentricity at the first available index.
                # This corresponds to the first extrema that occurs after the
                # initial time.
                measured_eccs[method] += [measured_ecc[0]]
                tmaxList[method] += [tref_out[-1]]
                tminList[method] += [tref_out[0]]
                # add measured ecc vs time plot for each method to corresponding axis
                ax = ax_measured_ecc_vs_time if len(methods) == 1 else ax_measured_ecc_vs_time[idx]
                # add only for idx0 that are multiples of args.slice. This is to reduce the
                # number of lines in the plot to make each line plot visible.
                if idx0 % args.slice == 0:
                    ax.plot(tref_out, measured_ecc, c=colors[idx0], label=f"{ecc:.7f}")
                if idx == len(methods) - 1:
                    ax.set_xlabel(r"$t$ [$M$]")
            except Exception:
                warnings.warn("Exception raised. Probably too small eccentricity "
                              "to detect any extrema.")
    # Iterate over methods to plots measured ecc vs eob eccs collected above
    # looping over all EOB eccs for each methods.
    for idx, method in enumerate(methods):
        # plot waveform eccs vs eob eccs for different methods
        ax_eob_vs_measured_ecc.loglog(
            model_eccs[method], measured_eccs[method], label=method,
            c=colorsDict.get(method, "C0"),
            ls=lstyles.get(method, "-"),
            lw=lwidths.get(method, 1),
            alpha=lalphas.get(method, 1),
            marker=None if args.paper else "."  # no marker for paper
        )
        # Customize the measured ecc vs time plots
        ax = ax_measured_ecc_vs_time if len(methods) == 1 else ax_measured_ecc_vs_time[idx]
        ax.grid()
        if len(tmaxList[method]) >= 1:
            tmin = max(tminList[method])  # choose the shortest
            tmax = max(tmaxList[method])
            ymax = max(measured_eccs[method])
            ymin = min(EOBeccs)
            ax.set_xlim(tmin, tmax)
            ax.set_ylim(ymin, ymax)
        ax.set_ylabel("$e$")
        ax.set_yscale("log")
        # add yticks
        ax.set_yticks(10.0**np.arange(-6.0, 1.0, 2.0))
        # add text indicating the method used
        ax.text(0.95, 0.95, method, ha="right", va="top",
                transform=ax.transAxes, fontsize=10)
        # add colorbar
        norm = mpl.colors.LogNorm(vmin=EOBeccs.min(), vmax=EOBeccs.max())
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig_measured_ecc_vs_time.colorbar(sm, cax=cax, orientation='vertical')
        # set yticks on colorbar
        cbar.ax.set_yticks(10**np.arange(-7.0, 1.0))
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label(r"$e_{\mathrm{EOB}}$ at $\omega_0$",
                       size=10)
        if idx == 0:
            ax.set_title(rf"$q={q:.1f}, \chi_{{1z}}={chi1z:.1f}, "
                         rf"\chi_{{2z}}={chi2z:.1f}$",
                         ha="center", fontsize=10)

    # Customize measured eccs vs eob eccs
    ax_eob_vs_measured_ecc.set_title(
        rf"$q$={q:.1f}, $\chi_{{1z}}$={chi1z:.1f}, $\chi_{{2z}}$"
        f"={chi2z:.1f}")
    ax_eob_vs_measured_ecc.legend(frameon=True)
    # set major ticks
    locmaj = mpl.ticker.LogLocator(base=10, numticks=20)
    ax_eob_vs_measured_ecc.xaxis.set_major_locator(locmaj)
    # set minor ticks
    locmin = mpl.ticker.LogLocator(base=10.0, subs=np.arange(0.1, 1.0, 0.1),
                                   numticks=20)
    ax_eob_vs_measured_ecc.xaxis.set_minor_locator(locmin)
    ax_eob_vs_measured_ecc.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    # set grid
    ax_eob_vs_measured_ecc.grid(which="major")
    ax_eob_vs_measured_ecc.set_xlabel(r"EOB Eccentricity $e_{\mathrm{EOB}}$")
    ax_eob_vs_measured_ecc.set_ylabel(r"Measured Eccentricity $e$")
    ax_eob_vs_measured_ecc.set_ylim(top=1.0)
    ax_eob_vs_measured_ecc.set_xlim(EOBeccs[0], EOBeccs[-1])

    # save figures
    fig_eob_vs_measured_ecc.savefig(
        f"{fig_eob_vs_measured_ecc_name}",
        bbox_inches="tight")
    fig_measured_ecc_vs_time.savefig(
        f"{fig_measured_ecc_vs_time_name}",
        bbox_inches="tight")

if "all" in args.param_set_key:
    args.param_set_key = list(available_param_sets.keys())

#  use fancy colors and other settings
journal = "APS" if args.paper else "Notebook"
use_fancy_plotsettings(journal=journal)

for key in args.param_set_key:
    plot_waveform_ecc_vs_model_ecc(args.method, key)
