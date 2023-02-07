"""Fancy settings for plots."""
from matplotlib import rc
from cycler import cycler
from matplotlib import colormaps

dark2 = colormaps["Dark2"].colors

colorsDict = {
    "default": dark2[1],  # brown
    "apocenter": dark2[3],  # dark2[2],  # purple
    "pericenter": "tab:blue",  # dark2[0],  # turquoise
    "vline": dark2[3],  # pink
    "hline": dark2[5],  # orange
    "edge": dark2[7],  # gray
    "pericentersvline": dark2[7],  # gray
    "FrequencyFits": dark2[2],
    "AmplitudeFits": dark2[3],
    "ResidualFrequency": dark2[5],
    "ResidualAmplitude": "tab:blue",  # dark2[0],
    "Frequency": dark2[0],
    "Amplitude": dark2[1],
    "turquoise": dark2[0],
    "brown": dark2[1],
    "purple": dark2[2],
    "pink": dark2[3],
    "olive": dark2[4],
    "orange": dark2[5],
    "gray": dark2[7],
    "blue": "tab:blue"
}

markersDict = {"apocenter": "s",
               "pericenter": "o",
               }
markersSizeDict = {"apocenter": 3,
                   "pericenter": 4,
                   }

lstyles = {"Amplitude": "solid",
           "Frequency": "dashdot",
           "ResidualAmplitude": "dashed",
           "ResidualFrequency": "solid",
           "FrequencyFits": "dashdot",
           "AmplitudeFits": "dotted"}
lwidths = {"Amplitude": 1,
           "Frequency": 1,
           "ResidualAmplitude": 1,
           "ResidualFrequency": 1,
           "FrequencyFits": 1,
           "AmplitudeFits": 3}
lalphas = {"Amplitude": 1,
           "Frequency": 1,
           "ResidualAmplitude": 1,
           "ResidualFrequency": 1,
           "FrequencyFits": 1,
           "AmplitudeFits": 1}

figWidthsOneColDict = {
    "APS": 3.4,
    "APJ": 3.543,
    "Elsevier": 3.543,
    "Springer": 3.3,
    "Presentation": 3,
    "Notebook": 6
}

figWidthsTwoColDict = {
    "APS": 7.0,
    "APJ": 7.48,
    "Elsevier": 7.48,
    "Springer": 6.93,
    "Presentation": 4.5,
    "Notebook": 12
}

figHeightsDict = {
    "APS": 2,
    "APJ": 2,
    "Elsevier": 2,
    "Springer": 2,
    "Presentation": 3.0,
    "Notebook": 4.0
}

ticklabelSizeDict = {"APS": 9.0,
                     "APJ": 8.0,
                     "Elsevier": 8.0,
                     "Springer": 8.0,
                     "Presentation": 8.0,
                     "Notebook": 14.0}
labelSizeDict = {"APS": 9.0,
                 "APJ": 8.0,
                 "Elsevier": 8.0,
                 "Springer": 8.0,
                 "Presentation": 8.0,
                 "Notebook": 18.0}
fontSizeDict = {"APS": 9.0,
                "APJ": 8.0,
                "Elsevier": 8.0,
                "Springer": 8.0,
                "Presentation": 8.0,
                "Notebook": 14.0}


def use_fancy_plotsettings(usetex=True, style="Notebook"):
    """Use fancy plot settings."""
    # Text
    if usetex:
        rc("text", usetex=usetex)
        rc("text.latex", preamble=r"\usepackage{amsmath}")
        rc("text.latex", preamble=r"\usepackage{txfonts}")
        rc("text.latex",
           preamble=r"\DeclareMathAlphabet{\mathpzc}{OT1}{pzc}{m}{it}")
    # Axes
    rc("axes", prop_cycle=cycler(color=dark2))  # color cycler
    rc("axes", linewidth=0.6)
    rc("axes", labelsize=labelSizeDict[style])
    rc("axes", titlesize=fontSizeDict[style])
    # Ticks
    rc("xtick", labelsize=ticklabelSizeDict[style])
    rc("ytick", labelsize=ticklabelSizeDict[style])
    rc("xtick", direction="in")
    rc("ytick", direction="in")
    # Legend
    rc("legend", frameon=False)
    rc("legend", fontsize=fontSizeDict[style])
    # Fonts
    rc("font", family="serif")
    rc("font", serif="times")
    rc("font", size=fontSizeDict[style])
    # Lines
    rc("lines", linewidth=1.0)


# Dictionary of labels to use in plots.
labelsDict = {
    "ecc": r"$e_{\mathrm{gw}}$",
    "mean_ano": r"$l_{\mathrm{gw}}$ [rad]",
    "eccentricity": r"$e_{\mathrm{gw}}$",
    "mean_anomaly": r"$l_{\mathrm{gw}}$ [rad]",
    "mean_anomaly_no_units": r"$l_{\mathrm{gw}}$",
    "omega22_pericenters": r"$\omega^{\mathrm{p}}_{22}$",
    "omega22_apocenters": r"$\omega^{\mathrm{a}}_{22}$",
    "omega22_average": r"$\langle\omega_{22}\rangle$",
    "omega22_average_dimless": r"$\langle\omega_{22}\rangle$ [rad/$M$]",
    "orbit_averaged_omega22_pericenters": r"$\langle\omega_{22}\rangle^{\mathrm{p}}$",
    "orbit_averaged_omega22_apocenters": r"$\langle\omega_{22}\rangle^{\mathrm{a}}$",
    "f22_average": r"$\langle f_{22}\rangle$",
    "f22_average_ref": r"$\langle f_{\mathrm{ref}}\rangle$",
    "f22_ref": r"$f_{\mathrm{ref}}$",
    "eob_eccentricity": r"$e_{\mathrm{eob}}$",
    "eob_mean_anomaly_no_units": r"$l_{\mathrm{eob}}$",
    "geodesic_eccentricity": r"$e_{\mathrm{geo}}$",
    "e_omega22": r"$e_{\omega_{22}}$",
    "t": r"$t$",
    "t_dimless": r"$t$ [$M$]",
    "t_start": r"$t_{0}$",
    "t_start_hat": r"$\widehat{t}_{0}$",
    "omega22_dimless": r"$\omega_{22}$ [rad/$M$]",
    "pericenters": "Pericenters",
    "apocenters": "Apocenters",
    "amp22": r"$A_{22}$",
    "amp22_fit": r"$A^{\mathrm{fit}}_{22}$",
    "omega22": r"$\omega_{22}$",
    "omega22_fit": r"$\omega^{\mathrm{fit}}_{22}$",
    "omega22_fit_pericenters": r"$\omega^{\mathrm{fit,p}}_{22}$",
    "omega22_fit_apocenters": r"$\omega^{\mathrm{fit,a}}_{22}$",
    "res_omega22": r"$\Delta\omega_{22}$",
    "res_omega22_dimless": r"$\Delta\omega_{22}$ [rad/$M$]",
    "res_amp22": r"$\Delta A_{22}$",
    "t_ref": r"$t_\mathrm{ref}$",
    "dedt": r"$de/dt$",
    "mean_of_interpolants": r"Mean of $\omega^{\mathrm{p}}_{22}$ and $\omega^{\mathrm{a}}_{22}$",
    "orbit_averaged_omega22": r"orbit averaged $\omega_{22}$",
    "omega22_zeroecc": r"$\omega_{22}$ of quasicircular counterpart",
    "omega_start": r"$\omega_0$",
    "q": r"$q$",
    "chi1z": r"$\chi_{1z}$",
    "chi2z": r"$\chi_{2z}$",
    "h22_real": r"Re[$\mathpzc{h}_{22}$]",
    "chirp_mass": r"$\mathcal{M}$"
}
