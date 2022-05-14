"""Fancy settings for plots."""
from matplotlib import rc
from cycler import cycler
from palettable.wesanderson import GrandBudapest3_6
from palettable.wesanderson import GrandBudapest4_5
from palettable.wesanderson import Darjeeling3_5

gb36 = GrandBudapest3_6.mpl_colors
gb45 = GrandBudapest4_5.mpl_colors
d35 = Darjeeling3_5.mpl_colors
colorList = [gb36[1], gb36[2], d35[4], gb36[4], gb36[5], gb45[1]]


def use_fancy_plotsettings():
    """Use fancy plot settings."""
    # LaTeX
    rc("text", usetex=True)
    # color cycler
    rc("axes", prop_cycle=cycler(color=colorList))
