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
    "peaksvline": colors_aq_15[1]
}


def use_fancy_plotsettings(usetex=True):
    """Use fancy plot settings."""
    # Text
    if usetex:
        rc("text", usetex=usetex)
        rc("text.latex", preamble=r"\usepackage{amsmath}")
        rc("text.latex", preamble=r"\usepackage{txfonts}")
    # Axes
    rc("axes", prop_cycle=cycler(color=colorsList))  # color cycler
    rc("axes", linewidth=0.6)
    rc("axes", labelsize=12)
    rc("axes", titlesize=12)
    # Ticks
    rc("xtick", labelsize=12)
    rc("ytick", labelsize=12)
    rc("xtick", direction="in")
    rc("ytick", direction="in")
    # Legend
    rc("legend", frameon=False)
    rc("legend", fontsize=12)
    # Fonts
    rc("font", family="serif")
    rc("font", serif="times")
