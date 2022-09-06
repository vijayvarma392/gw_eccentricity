"""Test smoothness of eccentricity definition methods using EMRI waveforms."""

__doc__ = """This test checks whether different eccentricity definitions vary
smoothly as a function of the internal eccentricity definition used by
Maarten's EMRI model. The EMRI waveforms that are used for this test are
generated from a fixed mass ratio, spins, and a fixed Momega0 (the initial
dimless orbital frequency) with eccentricity varying from 0 to 0.5. We try to
measure the eccentricity from these waveforms using different eccentricity
definitions. We test the smoothness of the measured eccentricities compared to
the EMRI eccentricities used by the waveform model to generate the waveforms in
the following two ways:

- For each waveform, we measure the eccentricity at the very first extrema
(pericenter or apocenter). That way, the measured eccentricity is also at
(nearly) Momega0. Finally, we plot the measured eccentricity vs the internal
EMRI eccentricity. You should check visually whether there are glitchy features
in these plots.

- For each waveform, We plot the measured eccentricity vs tref_out, and color
the lines by the input EMRI eccentricity at Momega0. You should check visually
whether there are glitchy features in these plots.

Usage: python emri_smoothness_test.py -d ecc_waveforms -m 'Amplitude' \
'ResidualAmplitude' -p 'all' python emri_smoothness_test.py -d ecc_waveforms -m
'all' -p 'all'
"""

import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
import pandas as pd
import glob
import re
sys.path.append("../../")
from gw_eccentricity import measure_eccentricity, get_available_methods
from gw_eccentricity.load_data import load_waveform
from gw_eccentricity.utils import SmartFormatter
from gw_eccentricity.plot_settings import use_fancy_plotsettings, figWidthsOneColDict
from gw_eccentricity.plot_settings import colorsDict, lwidths, lalphas, lstyles, labelsDict

parser = argparse.ArgumentParser(
    description=(__doc__),
    formatter_class=SmartFormatter)

parser.add_argument(
    "--data_dir", "-d",
    type=str,
    default="../../data/ecc_waveforms",
    help=("Base directory where waveform files are stored."))
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
          "Possible choices are 'all' OR one or more of 1, 2, 3.\n"
          "1: q=10, chi1z=chi2z=0.\n"
          "2: q=100, chi1z=chi2z=0.\n"
          "3: q=1000, chi1z=chi2z=0.\n"))
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
    "--paper",
    action="store_true",
    help="Remove markers for paper.")
parser.add_argument(
    "--debug_index",
    type=int,
    default=None,
    help=("Only analyse the waveform with this index, "
          "and enable debugging output"))
parser.add_argument(
    "--verbose",
    action='store_true',
    default=False,
    help="increase verbosity")

args = parser.parse_args()

# check that given method is available
for method in args.method:
    if method not in get_available_methods():
        raise KeyError(f"Method {method} is not an allowed method."
                       f" Must be one of {get_available_methods()}"
                       " or `all` for using all available methods.")
# Format: [q, chi1z, chi2z]
available_param_sets = {
    "1": [10, 0, 0],
    "2": [100, 0, 0],
    "3": [1000, 0, 0]
}
# check that given param set is available
for p in args.param_set_key:
    if p not in available_param_sets:
        raise KeyError(f"Param set key {p} is not an allowed param set key."
                       f" Must be one of {list(available_param_sets.keys())}"
                       " or `all` for using all available param set keys.")

data_dir = args.data_dir + "/Non-Precessing/EMRI/"

# set debug to False
# The default width is slightly larger than what we need.
# Setting it to smaller value helps finding all the extrema
extra_kwargs = {"debug": False,
                "extrema_finding_kwargs": {"width": 300}}

if args.debug_index is not None:
    extra_kwargs['debug'] = True

cmap = cm.get_cmap("plasma")


def get_file_names(key):
    """Get the list of file paths based on key."""
    q, chi1z, chi2z = available_param_sets[key]
    fileNames = sorted(
        glob.glob(f"{data_dir}/q{q}/EMRI_0PA_q{q}_e*[!_ecc].h5"))
    return fileNames


def get_color(ecc, eccs, colors):
    """Get color for the ecc vs time lines."""
    return colors[np.argmin(np.abs(eccs - ecc))]


def plot_waveform_ecc_vs_model_ecc(methods, key):
    """Create plots for given methods and parameter set."""
    # Get the output figure name
    # method_str is used in the filename for the output figure
    if "all" in methods:
        methods = get_available_methods()
        method_str = "all"
    else:
        method_str = "_".join(methods)

    if args.example:
        fig_ecc_at_start_name = (
            f"{args.fig_dir}/test_emri_vs_measured_ecc_example"
            f".{args.plot_format}")
        fig_ecc_vs_t_name = (
            f"{args.fig_dir}/test_emri_measured_ecc_vs_time_example."
            f"{args.plot_format}")
    else:
        fig_ecc_at_start_name = (
            f"{args.fig_dir}/test_emri_vs_measured_ecc_set{key}_{method_str}"
            f".{args.plot_format}")
        fig_ecc_vs_t_name = (
            f"{args.fig_dir}/test_emri_measured_ecc_vs_time_set{key}_{method_str}"
            f".{args.plot_format}")
    # For emri vs measured ecc at the first extrema
    fig_ecc_at_start, ax_ecc_at_start = plt.subplots(
        figsize=(figWidthsOneColDict[style], 2 if args.paper else 3))
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
    failed_eccs = {}  # To keep track of failed cases.
    failed_indices = {}  # To Keep track of failed indices
    # Initiate the dictionary with an empty list for each method
    for method in methods:
        model_eccs.update({method: []})
        measured_eccs_at_start.update({method: []})
        tmaxList.update({method: []})
        tminList.update({method: []})
        failed_eccs.update({method: []})
        failed_indices.update({method: []})
    q, chi1z, chi2z = available_param_sets[key]
    emri_ecc_label = labelsDict['omega_start']
    # We exclude the first file since it is the quasicircular waveform
    # We also exclude the last file since the waveform starts at lower
    # omega than others.
    filePaths = get_file_names(key)[1:-1]
    EMRIeccs = []
    for f in filePaths:
        ecc = float(re.findall("\d{1}\.\d{3}", f)[0])
        EMRIeccs.append(ecc)
    EMRIeccs = np.array(EMRIeccs)
    # Create an eccs array to map colors to eccentricities
    # for the ecc vs time line plots
    eccs_for_colors = np.linspace(
        np.round(EMRIeccs.min(), 2),
        np.round(EMRIeccs.max(), 2),
        len(filePaths))
    colors = cmap(np.linspace(0, 1, len(filePaths)))
    for idx0, f in tqdm(enumerate(filePaths), disable=args.verbose):
        # in debugging mode, skip all but debug_index:
        if args.debug_index is not None:
            if idx0 != args.debug_index:
                continue
        # Need to change the resolution of the data, otherwise the omega22
        # average are not monotonically increasing
        kwargs = {"filepath": f,
                  "deltaT": 0.08 if idx0 <= 8 else 0.05}
        if args.verbose:
            print(f"idx={idx0}, {f}")
        # check if any residual method is in methods. If yes then load
        # zeroecc waveforms also.
        if "ResidualAmplitude" in methods or "ResidualFrequency" in methods:
            kwargs.update({"include_zero_ecc": True})
        dataDict = load_waveform(origin="EMRI", **kwargs)
        tref_in = dataDict["t"]
        for idx, method in enumerate(methods):
            try:
                tref_out, measured_ecc, mean_ano = measure_eccentricity(
                    tref_in=tref_in,
                    dataDict=dataDict,
                    method=method,
                    extra_kwargs=extra_kwargs)
                model_eccs[method] += [EMRIeccs[idx0]]
                # Get the measured eccentricity at the first available index.
                # This corresponds to the first extrema that occurs after the
                # initial time
                measured_eccs_at_start[method] += [measured_ecc[0]]
                tmaxList[method] += [tref_out[-1]]
                tminList[method] += [tref_out[0]]
                # Add measured ecc vs time plot for each method to
                # corresponding axis.
                ax_ecc_vs_t[idx].plot(
                    tref_out,
                    measured_ecc,
                    c=get_color(EMRIeccs[idx0], eccs_for_colors, colors))
                if idx == len(methods) - 1:
                    ax_ecc_vs_t[idx].set_xlabel(labelsDict["t_dimless"])
            except Exception as e:
                # collected failures
                failed_eccs[method] += [EMRIeccs[idx0]]
                failed_indices[method] += [idx0]
                if args.debug_index is None:
                    # If verbose, print the exception message
                    if args.verbose:
                        print(f"Exception raised with the message {e}")
                else:
                    # if debugging, print entire traceback
                    raise
    # Iterate over methods to plots measured ecc at the first extrema vs emri
    # eccs collected above looping over all EMRI eccs for each methods
    for idx, method in enumerate(methods):
        # Plot measured eccs at the first extrema vs emri eccs for different
        # methods
        ax_ecc_at_start.plot(
            model_eccs[method], measured_eccs_at_start[method], label=method,
            c=colorsDict.get(method, "C0"),
            ls=lstyles.get(method, "-"),
            lw=lwidths.get(method, 1),
            alpha=lalphas.get(method, 1),
            marker=".",
            # marker=None if args.paper else "."  # No marker for paper
        )
        # Customize the measured ecc at the first extrema vs time plots
        ax_ecc_vs_t[idx].grid()
        if len(tmaxList[method]) >= 1:
            tmin = max(tminList[method])  # Choose the shortest
            tmax = max(tmaxList[method])
            ymax = max(measured_eccs_at_start[method])
            ymin = min(EMRIeccs)
            ax_ecc_vs_t[idx].set_xlim(tmin, tmax)
            ax_ecc_vs_t[idx].set_ylim(ymin, ymax)
        ax_ecc_vs_t[idx].set_ylabel(labelsDict["eccentricity"])
        # ax_ecc_vs_t[idx].set_yscale("log")
        # Add yticks.
        # ax_ecc_vs_t[idx].set_yticks(10.0**np.arange(-6.0, 1.0, 2.0))
        # Add text indicating the method used
        ax_ecc_vs_t[idx].text(0.95, 0.95, method, ha="right", va="top",
                              transform=ax_ecc_vs_t[idx].transAxes)
        # Add colorbar
        norm = mpl.colors.Normalize(vmin=eccs_for_colors.min(),
                                    vmax=eccs_for_colors.max())
        divider = make_axes_locatable(ax_ecc_vs_t[idx])
        cax = divider.append_axes('right', size='3%', pad=0.1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig_ecc_vs_t.colorbar(sm, cax=cax, orientation='vertical')
        # Set yticks on colorbar
        # cbar.ax.set_yticks(10**np.arange(-7.0, 1.0))
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label(
            rf"{labelsDict['geodesic_eccentricity']} at "
            rf"{emri_ecc_label}",
            size=10)
        if idx == 0:
            ax_ecc_vs_t[idx].set_title(
                rf"{labelsDict['q']}={q:.1f}, "
                fr"{labelsDict['chi1z']}={chi1z:.1f}, "
                rf"{labelsDict['chi2z']}={chi2z:.1f}",
                ha="center")

    # Customize measured eccs at the first extrema vs eob eccs
    ax_ecc_at_start.set_title(
        rf"{labelsDict['q']}={q:.1f}, "
        fr"{labelsDict['chi1z']}={chi1z:.1f}, "
        rf"{labelsDict['chi2z']}={chi2z:.1f}",
        ha="center")
    ax_ecc_at_start.legend(frameon=True)
    # set ticks
    ticks = np.arange(0, EMRIeccs.max(), 0.1)
    ticklabels = [f"{tick:.1f}" for tick in ticks]
    ax_ecc_at_start.set_xticks(ticks)
    ax_ecc_at_start.set_xticklabels(ticklabels)
    ax_ecc_at_start.set_yticks(ticks)
    ax_ecc_at_start.set_yticklabels(ticklabels)
    ax_ecc_at_start.set_xlabel(rf"{labelsDict['geodesic_eccentricity']} at "
                               rf"{emri_ecc_label}")
    ax_ecc_at_start.set_ylabel(labelsDict["eccentricity"])
    # ax_ecc_at_start.set_ylim(top=EMRIeccs.max())
    # ax_ecc_at_start.set_xlim(EMRIeccs.min(), EMRIeccs.max())
    ax_ecc_at_start.grid()

    # Save figures
    fig_ecc_at_start.savefig(
        f"{fig_ecc_at_start_name}",
        bbox_inches="tight")
    fig_ecc_vs_t.savefig(
        f"{fig_ecc_vs_t_name}",
        bbox_inches="tight")
    return failed_eccs, failed_indices


def report_failures(failed_eccs, failed_indices, methods):
    """Report failed cases."""
    for method in methods:
        num_failures = len(failed_eccs[method])
        print(f"================{method}============================")
        if num_failures == 0:
            print("All cases passed!")
        else:
            print(f"{num_failures} {'case' if num_failures == 1 else 'cases'} "
                  "failed.")
            df = pd.DataFrame({
                "Case indices": failed_indices[method],
                "Eccentricity": failed_eccs[method]
            })
            print(df)


if "all" in args.param_set_key:
    args.param_set_key = list(available_param_sets.keys())

#  Use fancy colors and other settings
style = "APS" if args.paper else "Notebook"
use_fancy_plotsettings(style=style)

for key in args.param_set_key:
    failed_eccs, failed_indices = plot_waveform_ecc_vs_model_ecc(
        args.method, key)
    print(f"================Test report for set {key}===================")
    report_failures(failed_eccs, failed_indices, args.method)
