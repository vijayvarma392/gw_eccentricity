"""
Find peaks and troughs using Residual Amplitude.

Part of Eccentricity Definition project.
Md Arif Shaikh, Mar 29, 2022
"""
from eccDefinition import eccDefinition
from utils import get_peak_via_quadratic_fit
from scipy.signal import find_peaks
from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np


class measureEccentricityUsingResidualAmplitude(eccDefinition):
    """Measure eccentricity by finding extrema from residual amplitude."""

    def __init__(self, dataDict):
        """Init for measureEccentricityWithResidualAmplitude class.

        parameters:
        ----------
        dataDict: Dictionary containing the eccentric and quasi
        circular waveform data.
        For residual amplitude method we need quasi-circular modes
        in additionn to the eccentric modes. Provide it as a dictionary
        for the key `hlm0` in the dataDict dictionary. Keys for the
        modes in the mode dictionary should be of the form `(l,m)`.
        """
        super().__init__(dataDict)
        self.hlm0 = self.dataDict["hlm0"]
        self.time0 = self.dataDict["t0"]
        self.h220 = self.hlm0[(2, 2)]
        self.time0 = self.time0 - get_peak_via_quadratic_fit(
            self.time0,
            self.h220)[0]
        self.quasi_circ_amp_interp = InterpolatedUnivariateSpline(
            self.time0, np.abs(self.h220))
        self.res_amp22 = self.amp22 - self.quasi_circ_amp_interp(self.time)

    def find_extrema(self, which="maxima", height=None, threshold=None,
                     distance=None, prominence=None, width=10, wlen=None,
                     rel_height=0.5, plateau_size=None):
        """Find the extrema in the residual amplitude from 22 mode.

        parameters:
        -----------
        which: either maxima or minima
        see scipy.signal.find_peaks for rest or the arguments.

        returns:
        ------
        array of positions of extrema in residual amplitude.
        """
        if which == "maxima" or which == "peaks":
            sign = 1
        elif which == "minima" or which == "troughs":
            sign = - 1
        else:
            raise Exception("`which` must be one of ['maxima', 'minima',"
                            " 'peaks', 'troughs']")
        return find_peaks(sign * self.res_amp22, height, threshold, distance,
                          prominence, width, wlen, rel_height, plateau_size)[0]
