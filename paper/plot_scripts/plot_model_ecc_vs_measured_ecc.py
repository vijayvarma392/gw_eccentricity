"""Plot model ecc vs measured ecc."""

import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
sys.path.append("../../")
from measureEccentricity.plot_settings import figWidthsOneColDict, use_fancy_plotsettings, colorsDict

data = pd.read_csv("../../data/model_ecc_vs_measured_ecc.csv")

q = 1
chi1 = 0
chi2 = 0

use_fancy_plotsettings(journal="APS")

fig, ax = plt.subplots(figsize=(figWidthsOneColDict["APS"], 3))
ax.loglog(data["model_ecc"], data["SEOB_measured_ecc"], marker=".",
          label="SEOBNRv4EHM", c=colorsDict["apastron"], markersize=3)
ax.loglog(data["model_ecc"], data["TEOB_measured_ecc"], marker=".",
          label="TEOBResumS", c=colorsDict["periastron"], markersize=3)
ax.loglog(data["model_ecc"], data["SEOBNRE_measured_ecc"], marker=".",
          label="SEOBNRE", c=colorsDict["vline"], markersize=3)

ax.legend(frameon=True)
# set major ticks
locmaj = mpl.ticker.LogLocator(base=10, numticks=20)
ax.xaxis.set_major_locator(locmaj)
# set minor ticks
locmin = mpl.ticker.LogLocator(base=10.0,
                               subs=np.arange(0.1, 1.0, 0.1),
                               numticks=20)
ax.xaxis.set_minor_locator(locmin)
ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
# set grid
ax.grid(which="major")
ax.set_xlabel(r"Model Eccentricity")
ax.set_ylabel(r"Measured Eccentricity $e$")
ax.set_ylim(bottom=1e-6, top=1.0)
ax.set_xlim(min(data["model_ecc"]), 1.0)
ax.set_title(rf"$q={q:.1f}$, $\chi_{{1z}}={chi1:.1f}$, "
             rf"$\chi_{{2z}}={chi2:.1f}$", ha="center", fontsize=9)
fig.savefig("../../paper/figs/model_ecc_vs_measured_ecc_set1.pdf",
            bbox_inches="tight")
