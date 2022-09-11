"""Fancy settings for plots."""
from matplotlib import rc
from cycler import cycler
from palettable.wesanderson import Aquatic1_5
from palettable.wesanderson import Darjeeling2_5
from palettable.wesanderson import Darjeeling3_5
from palettable.wesanderson import FantasticFox2_5
from palettable.wesanderson import GrandBudapest5_5
from palettable.wesanderson import GrandBudapest1_4
from palettable.wesanderson import GrandBudapest4_5
from palettable.wesanderson import GrandBudapest3_6
from palettable.wesanderson import Zissou_5
from palettable.wesanderson import Royal1_4
colors_aq_15 = Aquatic1_5.mpl_colors
colors_dj_25 = Darjeeling2_5.mpl_colors
colors_dj_35 = Darjeeling3_5.mpl_colors
colors_ff_25 = FantasticFox2_5.mpl_colors
colors_gb_55 = GrandBudapest5_5.mpl_colors
colors_gb_14 = GrandBudapest1_4.mpl_colors
colors_gb_45 = GrandBudapest4_5.mpl_colors
colors_gb_36 = GrandBudapest3_6.mpl_colors
colors_zs_5 = Zissou_5.mpl_colors
colors_ry_14 = Royal1_4.mpl_colors

colorsList = [colors_gb_14[1],
              colors_gb_36[1],
              colors_gb_45[0],
              colors_gb_55[1],
              colors_dj_25[1]]

colorsDict = {
    "default": colors_dj_25[3],  # colors_gb_36[5],
    "apocenter": colors_dj_25[1],
    "pericenter": colors_gb_45[1],
    "vline": colors_gb_14[1],
    "pericentersvline": colors_aq_15[1],
    "FrequencyFits": colors_dj_25[1],
    "AmplitudeFits": colors_gb_55[1],
    "ResidualFrequency": colors_gb_14[1],
    "ResidualAmplitude": colors_gb_45[1],
    "Frequency": colors_aq_15[1],
    "Amplitude": colors_dj_25[3]  # colors_gb_36[5],
}

lstyles = {"Amplitude": "solid",
           "Frequency": "dashdot",
           "ResidualAmplitude": "dashed",
           "ResidualFrequency": "solid",
           "FrequencyFits": "dotted",
           "AmplitudeFits": "dashdot"}
lwidths = {"Amplitude": 1,
           "Frequency": 1,
           "ResidualAmplitude": 1,
           "ResidualFrequency": 1,
           "FrequencyFits": 3,
           "AmplitudeFits": 1}
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
    rc("axes", prop_cycle=cycler(color=colorsList))  # color cycler
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
    "omega22_pericenters": r"$\omega^{\mathrm{p}}_{22}$",
    "omega22_apocenters": r"$\omega^{\mathrm{a}}_{22}$",
    "omega22_average": r"$\langle\omega_{22}\rangle$",
    "omega22_average_dimless": r"$\langle\omega_{22}\rangle$ [rad/$M$]",
    "eob_eccentricity": r"$e_{\mathrm{eob}}$",
    "geodesic_eccentricity": r"$e_{\mathrm{g}}$",
    "e_omega22": r"$e_{\omega_{22}}$",
    "t": r"$t$",
    "t_dimless": r"$t$ [$M$]",
    "t_start": r"$t_{0}$",
    "omega22_dimless": r"$\omega_{22}$ [rad/$M$]",
    "pericenters": "Pericenters",
    "apocenters": "Apocenters",
    "amp22": r"$A_{22}$",
    "omega22": r"$\omega_{22}$",
    "res_omega22": r"$\Delta\omega_{22}$",
    "res_omega22_dimless": r"$\Delta\omega_{22}$ [rad/$M$]",
    "res_amp22": r"$\Delta A_{22}$",
    "t_ref": r"$t_\mathrm{ref}$",
    "dedt": r"$de/dt$",
    "mean_of_interpolants": r"Mean of $\omega^{\mathrm{p}}_{22}$ and $\omega^{\mathrm{a}}_{22}$",
    "mean_motion": r"Mean motion",
    "omega22_zeroecc": r"$\omega_{22}$ of quasicircular counterpart",
    "omega_start": r"$\omega_0$",
    "q": r"$q$",
    "chi1z": r"$\chi_{1z}$",
    "chi2z": r"$\chi_{2z}$",
    "h22_real": r"Re[$\mathpzc{h}_{22}$]"
}
