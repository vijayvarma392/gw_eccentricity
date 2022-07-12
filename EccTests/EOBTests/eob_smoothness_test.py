"""Test smoothness of eccentricity definition methods using EOB waveforms."""

__doc__ = """This test checks whether different eccentricity definitions vary
smoothly as a function of the internal eccentricity definition used by Toni's
EOB model. The EOB waveforms that are used for this test are generated from a
fixed mass ratio, spins, and Momega0 (the initial dimless orbital frequency),
with eccentricity varying from 1e-7 to 1. We try to measure the eccentricity
from these waveforms using different eccentricity definitions. We test the
smoothness of the measured eccentricities compared to the eob eccentricities
used by the waveform model to generate the waveforms in the following two ways:

- For each waveform, we measure the eccentricity at the very first extrema
(pericenter or apocenter). That way, the measured eccentricity is also at
(nearly) Momega0. Finally, we plot the measured eccentricity vs the internal
EOB eccentricity.  You should check visually whether there are glitchy features
in these plots.

- For each waveform, We plot the measured eccentricity vs tref_out, and color
the lines by the input EOB eccentricity at Momega0. You should check visually
whether there are glitchy features in these plots.

Usage:
python eob_smoothness_test.py -d ecc_waveforms -m 'Amplitude' \
'ResidualAmplitude' -p 'all'
python eob_smoothness_test.py -d ecc_waveforms -m 'all' -p 'all'
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
    help=("Slice the eccentricities to loop over (EOBecccs array containing"
          " 150 eccs from 1e-7 to 1.0) by taking only every nth ecc value"
          " given by slice. This is useful when we do not want to plot"
          " for all the ecc but skip n number of ecc given by slice."
          " NOTE: This is applied only for the measured ecc vs time plot to"
          " make each line readable. Specially for presenting in the paper."
          " For regular test using the default value, i.e.,"
          " no slicing is recommended."))
parser.add_argument(
    "--paper",
    action="store_true",
    help="Remove markers for paper.")
parser.add_argument("--debug_index", type=int, default=None,
                    help="Only analyse the waveform with this index, and enable debugging output")
parser.add_argument("--verbose", action='store_true', default=False,
                    help="increase verbosity")

args = parser.parse_args()

EOBeccs = 10**np.linspace(-7, 0, 150)
Momega0 = 0.01
# We generate the waveforms with initial mean anomaly = pi/2 so that over the
# entire range of initial eccentricities 1e-7 to 1, the length of generated
# waveforms (inspiral time) does not change too much which is the case if we
# start with mean anomaly = 0. This is because at mean_ano = pi/2 the
# instantaneous frequency is close to the orbit averaged frequency.
meanAno = np.pi/2
Momega0_zeroecc = 0.002
# Format: [q, chi1z, chi2z]
available_param_sets = {
    "1": [1, 0, 0],
    "2": [2, 0.5, 0.5],
    "3": [4, -0.6, -0.6],
    "4": [6, 0.4, -0.4]}
data_dir = args.data_dir + "/Non-Precessing/EOB/"

# set debug to False
extra_kwargs = {"debug": False,
                # "treat_mid_points_between_peaks_as_troughs": True
                }

if not args.debug_index is None:
    extra_kwargs['debug']=True 

cmap = cm.get_cmap("plasma")
colors = cmap(np.linspace(0, 1, len(EOBeccs)))


def plot_waveform_ecc_vs_model_ecc(methods, key):
    """Create plots for given methods and parameter set."""
    # Get the output figure name
    # method_str is used in the filename for the output figure
    if "all" in methods:
        methods = get_available_methods()
        method_str = "all"
    else:
        method_str = "_".join(args.method)

    if args.example:
        fig_ecc_at_start_name = (
            f"{args.fig_dir}/test_eob_vs_measured_ecc_example"
            f".{args.plot_format}")
        fig_ecc_vs_t_name = (
            f"{args.fig_dir}/test_measured_ecc_vs_time_example."
            f"{args.plot_format}")
    else:
        fig_ecc_at_start_name = (
            f"{args.fig_dir}/test_eob_vs_measured_ecc_set{key}_{method_str}"
            f".{args.plot_format}")
        fig_ecc_vs_t_name = (
            f"{args.fig_dir}/test_measured_ecc_vs_time_set{key}_{method_str}"
            f".{args.plot_format}")
    # For eob vs measured ecc at the first extrema
    fig_ecc_at_start, ax_ecc_at_start = plt.subplots(
        figsize=(figWidthsOneColDict[style], 3))
    # For measured ecc vs time
    fig_ecc_vs_t, ax_ecc_vs_t = plt.subplots(
        nrows=len(methods),
        figsize=(figWidthsOneColDict[style],  # Figure width
                 (2 if args.paper else 3)  # Height of each row
                 * len(methods)  # Number of rows
                 ))
    if len(methods) == 1:
        ax_ecc_vs_t = [ax_ecc_vs_t]
    # Create dictionaries to store different quantities for each methods
    # as we loop over all the eccentricity and methods.
    model_eccs = {}  # To keep track of eob input eccentricity.
    measured_eccs_at_start = {}  # To keep track of measured ecc at the first
    # extrema.
    tmaxList = {}  # To keep track of minimum time in tref_out across all eccs.
    tminList = {}  # To keep track of maximum time in tref_out across all eccs.
    # Initiate the dictionary with an empty list for each method
    for method in methods:
        model_eccs.update({method: []})
        measured_eccs_at_start.update({method: []})
        tmaxList.update({method: []})
        tminList.update({method: []})
    q, chi1z, chi2z = available_param_sets[key]
    for idx0, ecc in tqdm(enumerate(EOBeccs), disable=args.verbose):

        # in debugging mode, skip all but debug_index:
        if not args.debug_index is None:
            if idx0!=args.debug_index: continue 

        fileName = (f"{data_dir}/EccTest_q{q:.2f}_chi1z{chi1z:.2f}_"
                    f"chi2z{chi2z:.2f}_EOBecc{ecc:.10f}_"
                    f"Momega0{Momega0:.3f}_meanAno{meanAno:.3f}.h5")
        kwargs = {"filepath": fileName}
        if args.verbose:
            print(f"idx={idx0}, {fileName}")
        # check if any residual method is in methods. If yes then load
        # zeroecc waveforms also.
        if "ResidualAmplitude" in methods or "ResidualFrequency" in methods:
            fileName_zero_ecc = (
                f"{data_dir}/EccTest_q{q:.2f}_chi1z"
                f"{chi1z:.2f}_"
                f"chi2z{chi2z:.2f}_EOBecc{0:.10f}_"
                f"Momega0{Momega0_zeroecc:.3f}_meanAno{meanAno:.3f}.h5")
            kwargs.update({"filepath_zero_ecc": fileName_zero_ecc,
                           "include_zero_ecc": True})
        dataDict = load_waveform(catalog="EOB", **kwargs)
        tref_in = dataDict["t"]
        for idx, method in enumerate(methods):
            try:
                tref_out, measured_ecc, mean_ano = measure_eccentricity(
                    tref_in=tref_in,
                    dataDict=dataDict,
                    method=method,
                    extra_kwargs=extra_kwargs)
                model_eccs[method] += [ecc]
                # Get the measured eccentricity at the first available index.
                # This corresponds to the first extrema that occurs after the
                # initial time
                measured_eccs_at_start[method] += [measured_ecc[0]]
                tmaxList[method] += [tref_out[-1]]
                tminList[method] += [tref_out[0]]
                # Add measured ecc vs time plot for each method to
                # corresponding axis.  Add only for idx0 that are multiples of
                # args.slice. This is to reduce the number of lines in the plot
                # to make each line plot visible. This is true only for this
                # plot.  For ecc at first extrema vs eob eccs, we plot for all
                # eccs.
                if idx0 % args.slice == 0:
                    ax_ecc_vs_t[idx].plot(tref_out, measured_ecc,
                                          c=colors[idx0], label=f"{ecc:.7f}")
                if idx == len(methods) - 1:
                    ax_ecc_vs_t[idx].set_xlabel(r"$t$ [$M$]")
            except Exception:
                if args.debug_index is None:
                    warnings.warn("Exception raised. Probably too small"
                                  " eccentricity to detect any extrema.")
                else:
                    raise  # if debugging, print entire traceback
    # Iterate over methods to plots measured ecc at the first extrema vs eob
    # eccs collected above looping over all EOB eccs for each methods
    for idx, method in enumerate(methods):
        # Plot measured eccs at the first extrema vs eob eccs for different
        # methods
        ax_ecc_at_start.loglog(
            model_eccs[method], measured_eccs_at_start[method], label=method,
            c=colorsDict.get(method, "C0"),
            ls=lstyles.get(method, "-"),
            lw=lwidths.get(method, 1),
            alpha=lalphas.get(method, 1),
            marker=None if args.paper else "."  # No marker for paper
        )
        # Customize the measured ecc at the first extrema vs time plots
        ax_ecc_vs_t[idx].grid()
        if len(tmaxList[method]) >= 1:
            tmin = max(tminList[method])  # Choose the shortest
            tmax = max(tmaxList[method])
            ymax = max(measured_eccs_at_start[method])
            ymin = min(EOBeccs)
            ax_ecc_vs_t[idx].set_xlim(tmin, tmax)
            ax_ecc_vs_t[idx].set_ylim(ymin, ymax)
        ax_ecc_vs_t[idx].set_ylabel("$e$")
        ax_ecc_vs_t[idx].set_yscale("log")
        # Add yticks.
        ax_ecc_vs_t[idx].set_yticks(10.0**np.arange(-6.0, 1.0, 2.0))
        # Add text indicating the method used
        ax_ecc_vs_t[idx].text(0.95, 0.95, method, ha="right", va="top",
                              transform=ax_ecc_vs_t[idx].transAxes,
                              fontsize=10)
        # Add colorbar
        norm = mpl.colors.LogNorm(vmin=EOBeccs.min(), vmax=EOBeccs.max())
        divider = make_axes_locatable(ax_ecc_vs_t[idx])
        cax = divider.append_axes('right', size='3%', pad=0.1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig_ecc_vs_t.colorbar(sm, cax=cax, orientation='vertical')
        # Set yticks on colorbar
        cbar.ax.set_yticks(10**np.arange(-7.0, 1.0))
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label(r"$e_{\mathrm{EOB}}$ at $\omega_0$",
                       size=10)
        if idx == 0:
            ax_ecc_vs_t[idx].set_title(
                rf"$q={q:.1f}, \chi_{{1z}}={chi1z:.1f}, "
                rf"\chi_{{2z}}={chi2z:.1f}$",
                ha="center", fontsize=10)

    # Customize measured eccs at the first extrema vs eob eccs
    ax_ecc_at_start.set_title(
        rf"$q$={q:.1f}, $\chi_{{1z}}$={chi1z:.1f}, $\chi_{{2z}}$"
        f"={chi2z:.1f}")
    ax_ecc_at_start.legend(frameon=True)
    # Set major ticks
    locmaj = mpl.ticker.LogLocator(base=10, numticks=20)
    ax_ecc_at_start.xaxis.set_major_locator(locmaj)
    # Set minor ticks
    locmin = mpl.ticker.LogLocator(base=10.0, subs=np.arange(0.1, 1.0, 0.1),
                                   numticks=20)
    ax_ecc_at_start.xaxis.set_minor_locator(locmin)
    ax_ecc_at_start.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    # Set grid
    ax_ecc_at_start.grid(which="major")
    ax_ecc_at_start.set_xlabel(r"EOB Eccentricity $e_{\mathrm{EOB}}$")
    ax_ecc_at_start.set_ylabel(r"Measured Eccentricity $e$")
    ax_ecc_at_start.set_ylim(top=1.0)
    ax_ecc_at_start.set_xlim(EOBeccs[0], EOBeccs[-1])

    # Save figures
    fig_ecc_at_start.savefig(
        f"{fig_ecc_at_start_name}",
        bbox_inches="tight")
    fig_ecc_vs_t.savefig(
        f"{fig_ecc_vs_t_name}",
        bbox_inches="tight")


if "all" in args.param_set_key:
    args.param_set_key = list(available_param_sets.keys())

#  Use fancy colors and other settings
style = "APS" if args.paper else "Notebook"
use_fancy_plotsettings(style=style)

for key in args.param_set_key:
    plot_waveform_ecc_vs_model_ecc(args.method, key)
