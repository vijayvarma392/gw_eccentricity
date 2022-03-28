"""
Base module to measure eccentricity and mean anomaly for given waveform data.

Part of Defining eccentricity project
Md Arif Shaikh, Mar 28, 2022
"""

import numpy as np


class eccDefinition:
    """Measure eccentricity from given waveform data dictionary."""

    def __init__(self, dataDict):
        """Init eccDefinition class.

        parameters:
        ----------
        datadict: Dictionary conntaining time, modes, etc
        """
        self.dataDict = dataDict
        self.time = self.dataDict["t"]
        self.h22 = self.dataDict["(2,2)"]
        self.amp22 = np.abs(self.h22)
        self.phase22 = - np.unwrap(np.angle(self.h22))
        self.omega22 = np.gradient(self.phase22, self.time)

    def find_peaks(self, order=10):
        """Find the peaks in the data.

        parameters:
        -----------
        order: window/width of peaks
        """
        "Not Implemented. Override it."

    def find_troughs(self, order=10):
        """Find the troughs in the data.

        parameters:
        -----------
        order: window/width of troughs
        """
        "Not Implemented. Override it."

    def peaks_interp(self, order=10, **kwargs):
        """Interpolator through peaks."""

    def troughs_interp(self, order=10, **kwargs):
        """Interpolator through troughs."""

    def measure_ecc(self, t_ref, order=10, **kwargs):
        """Measure eccentricity and meann anomaly at reference time.

        parameters:
        ----------
        t_ref: reference time to measure eccentricity and mean anomaly.
        order: width of peaks/troughs
        kwargs: any extra kwargs to the peak/trough findining functions.
        """
