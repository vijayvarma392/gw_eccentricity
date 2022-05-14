"""Fancy settings for plots."""
from matplotlib import rc
from cycler import cycler

use_palettable = False
try:
    import palettable
    use_palettable = True
except ImportError:
    print("palettable not found, using regular old colors. Get palettable"
          + " with 'pip install palettable' to get fancy colors.")

if use_palettable:
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

    colors_dict = {
            'BhA_traj': colors_zs_5[2],
            'BhB_traj': colors_gb_14[2],
            'BhA_spin': colors_gb_45[3],
            'BhB_spin': colors_aq_15[1],
            'BhC_spin': colors_zs_5[0],
            'L': colors_gb_55[1],
            'J': colors_aq_15[0],
            'info': colors_aq_15[0],
            'h+': colors_dj_25[3],
            'hx': colors_dj_25[1],
            }
else:
    colors_dict = {
            'BhA_traj': 'indianred',
            'BhB_traj': 'rebeccapurple',
            'BhA_spin': 'goldenrod',
            'BhB_spin': 'steelblue',
            'BhC_spin': 'forestgreen',
            'L': 'orchid',
            'info': 'k',
            'h+': 'tomato',
            'hx': 'steelblue',
            }


def use_fancy_plotsettings():
    """Use fancy plot settings."""
    # LaTeX
    rc("text", usetex=True)
    # color cycler
    colorList = [colors_dict[key] for key in colors_dict]
    rc("axes", prop_cycle=cycler(color=colorList))
