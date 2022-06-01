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
    "default": colors_gb_36[5],
    "apastron": colors_dj_25[1],
    "periastron": colors_gb_45[1],
    "vline": colors_gb_14[1],
    "peaksvline": colors_aq_15[1],
    "FrequencyFits": colors_dj_25[1],
    "ResidualFrequency": colors_gb_14[1],
    "ResidualAmplitude": colors_gb_45[1],
    "Frequency": colors_aq_15[1],
    "Amplitude": colors_gb_36[5],
}

lstyles = {"Amplitude": "solid",
           "Frequency": "dashdot",
           "ResidualAmplitude": "dashed",
           "ResidualFrequency": "solid",
           "FrequencyFits": "dotted"}
lwidths = {"Amplitude": 1,
           "Frequency": 1,
           "ResidualAmplitude": 1,
           "ResidualFrequency": 1,
           "FrequencyFits": 3}
lalphas = {"Amplitude": 1,
           "Frequency": 1,
           "ResidualAmplitude": 1,
           "ResidualFrequency": 1,
           "FrequencyFits": 1}

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
    "Notebook": 10
}

labelSizeDict = {"APS": 9.0,
                 "APJ": 8.0,
                 "Elsevier": 8.0,
                 "Springer": 8.0,
                 "Presentation": 8.0,
                 "Notebook": 12.0}
fontSizeDict = {"APS": 9.0,
                "APJ": 8.0,
                "Elsevier": 8.0,
                "Springer": 8.0,
                "Presentation": 8.0,
                "Notebook": 12.0}


def use_fancy_plotsettings(usetex=True, journal="Notebook"):
    """Use fancy plot settings."""
    # Text
    if usetex:
        rc("text", usetex=usetex)
        rc("text.latex", preamble=r"\usepackage{amsmath}")
        rc("text.latex", preamble=r"\usepackage{txfonts}")
    # Axes
    rc("axes", prop_cycle=cycler(color=colorsList))  # color cycler
    rc("axes", linewidth=0.6)
    rc("axes", labelsize=labelSizeDict[journal])
    rc("axes", titlesize=labelSizeDict[journal])
    # Ticks
    rc("xtick", labelsize=labelSizeDict[journal])
    rc("ytick", labelsize=labelSizeDict[journal])
    rc("xtick", direction="in")
    rc("ytick", direction="in")
    # Legend
    rc("legend", frameon=False)
    rc("legend", fontsize=labelSizeDict[journal])
    # Fonts
    rc("font", family="serif")
    rc("font", serif="times")
