"""
Base module to measure eccentricity and mean anomaly for given waveform data.

Part of Defining eccentricity project
Md Arif Shaikh, Mar 29, 2022
"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import lal
import lalsimulation as lalsim


class eccDefinition:
    """Measure eccentricity from given waveform data dictionary."""

    def __init__(self, dataDict):
        """Init eccDefinition class.

        parameters:
        ----------
        dataDict: Dictionary conntaining time, modes, etc
        """
        self.dataDict = dataDict
        self.time = self.dataDict["t"]
        self.hlm = self.dataDict["hlm"]
        self.h22 = self.hlm[(2, 2)]
        self.amp22 = np.abs(self.h22)
        self.time = self.time - get_peak_via_quadratic_fit(
            self.time, self.amp22)[0]
        self.phase22 = - np.unwrap(np.angle(self.h22))
        self.omega22 = np.gradient(self.phase22, self.time)

    def find_peaks(self, order=10):
        """Find the peaks in the data.

        parameters:
        -----------
        order: window/width of peaks
        """
        raise NotImplementedError("Please override me.")

    def find_troughs(self, order=10):
        """Find the troughs in the data.

        parameters:
        -----------
        order: window/width of troughs
        """
        raise NotImplementedError("Please override me.")

    def peaks_interp(self, order=10, **kwargs):
        """Interpolator through peaks."""
        peak_idx = self.find_peaks(order)
        if len(peak_idx) >= 2:
            return InterpolatedUnivariateSpline(self.time[peak_idx],
                                                self.omega22[peak_idx],
                                                **kwargs)
        else:
            print("...Number of peaks is less than 2. Not able"
                  " to create an interpolator.")
            return None

    def troughs_interp(self, order=10, **kwargs):
        """Interpolator through troughs."""
        trough_idx = self.find_troughs(order)
        if len(trough_idx) >= 2:
            return InterpolatedUnivariateSpline(self.time[trough_idx],
                                                self.omega22[trough_idx],
                                                **kwargs)
        else:
            print("...Number of troughs is less than 2. Not able"
                  " to create an interpolator.")
            return None

    def measure_ecc(self, t_ref, order=10, **kwargs):
        """Measure eccentricity and mean anomaly at reference time.

        parameters:
        ----------
        t_ref: reference time to measure eccentricity and mean anomaly.
        order: width of peaks/troughs
        kwargs: any extra kwargs to the peak/trough findining functions.

        returns:
        --------
        ecc_ref: measured eccentricity at t_ref
        mean_ano_ref: measured mean anomaly at t_ref
        """
        default_kwargs = {"w": None,
                          "bbox": [None, None],
                          "k": 3,
                          "ext": 0,
                          "check_finite": False}
        for kw in default_kwargs.keys():
            if kw in kwargs:
                default_kwargs[kw] = kwargs[kw]

        peaks_interpolator = self.peaks_interp(order, **default_kwargs)
        troughs_interpolator = self.troughs_interp(order, **default_kwargs)

        if peaks_interpolator is None or troughs_interpolator is None:
            print("...Sufficient number of peaks/troughs are not found."
                  " Can not create an interpolator. Most probably the "
                  "excentricity is too small. Returning eccentricity to be"
                  " zero")
            ecc_ref = 0
            mean_ano_ref = 0
        else:
            eccVals = ((np.sqrt(np.abs(peaks_interpolator(self.time)))
                        - np.sqrt(np.abs(troughs_interpolator(self.time))))
                       / (np.sqrt(np.abs(peaks_interpolator(self.time)))
                          + np.sqrt(np.abs(troughs_interpolator(self.time)))))
            ecc_interpolator = InterpolatedUnivariateSpline(self.time, eccVals,
                                                            **default_kwargs)
            ecc_ref = ecc_interpolator(t_ref)

            idx_peaks = self.find_peaks(order)
            meanperAnoVals = np.array([])
            timeVals = np.array([])

            idx = 0
            while idx < len(idx_peaks) - 1:
                orbital_period = (self.time[idx_peaks[idx + 1]]
                                  - self.time[idx_peaks[idx]])
                time_since_last_peak = (
                    self.time[idx_peaks[idx]: idx_peaks[idx + 1]]
                    - self.time[idx_peaks[idx]])
                meanperAno = 2 * np.pi * time_since_last_peak / orbital_period
                meanperAnoVals = np.append(meanperAnoVals, meanperAno)
                timeVals = np.append(
                    timeVals,
                    self.time[idx_peaks[idx]: idx_peaks[idx + 1]])
                idx += 1

            mean_ano_interpolator = InterpolatedUnivariateSpline(
                timeVals, meanperAnoVals, **kwargs)
            mean_ano_ref = mean_ano_interpolator(t_ref)

        return ecc_ref, mean_ano_ref


def get_peak_via_quadratic_fit(t, func):
    """
    Find the peak time of a function quadratically.

    Fits the function to a quadratic over the 5 points closest to the argmax
    func.
    t : an array of times
    func : array of function values
    Returns: tpeak, fpeak
    """
    # Find the time closest to the peak, making sure we have room on either
    # side
    index = np.argmax(func)
    index = max(2, min(len(t) - 3, index))

    # Do a quadratic fit to 5 points,
    # subtracting t[index] to make the matrix inversion nice
    testTimes = t[index-2:index+3] - t[index]
    testFuncs = func[index-2:index+3]
    xVecs = np.array([np.ones(5), testTimes, testTimes**2.])
    invMat = np.linalg.inv(np.array([[v1.dot(v2) for v1 in xVecs]
                                     for v2 in xVecs]))

    yVec = np.array([testFuncs.dot(v1) for v1 in xVecs])
    coefs = np.array([yVec.dot(v1) for v1 in invMat])
    return t[index] - coefs[1]/(2.*coefs[2]), (coefs[0]
                                               - coefs[1]**2./4/coefs[2])


def generate_waveform(approximant, q, chi1, chi2, deltaTOverM, Momega0,
                      inclination=0, phi_ref=0., longAscNodes=0,
                      eccentricity=0, meanPerAno=0,
                      alignedSpin=True, lambda1=None, lambda2=None):
    """Generate waveform for a given approximant using LALSuite.

    Returns dimless time and dimless complex strain.
    parameters:
    ----------
    approximant     # str, name of approximant
    q               # float, mass ratio q>=1
    chi1            # array/list of len=3, dimensionless spin vector of larger BH
    chi2            # array/list of len=3, dimensionless spin vector of smaller BH
    deltaTOverM     # float, dimensionless time step size
    Momega0          # float, dimensionless starting orbital frequency for waveform (rad/s)
    inclination     # float, inclination angle in radians
    phi_ref         # float, lalsim stuff
    longAscNodes    # float, Longiture of Ascending nodes
    eccentricity    # float, Eccentricity
    meanPerAno      # float, Mean anomaly of periastron
    alignedSpin     # assume aligned spin approximant
    lambda1         # tidal parameter for larger BH
    lambda2         # tidal parameter for smaller BH

    return:
    t               # array, dimensionless time
    h               # complex array, dimensionless complex strain h_{+} -i*h_{x}
    """
    chi1 = np.array(chi1)
    chi2 = np.array(chi2)

    if alignedSpin:
        if np.sum(np.sqrt(chi1[:2]**2)) > 1e-5 or np.sum(np.sqrt(chi2[:2]**2)) > 1e-5:
            raise Exception("Got precessing spins for aligned spin "
                            "approximant.")
        if np.sum(np.sqrt(chi1[:2]**2)) != 0:
            chi1[:2] = 0
        if np.sum(np.sqrt(chi2[:2]**2)) != 0:
            chi2[:2] = 0

    # sanity checks
    if np.sqrt(np.sum(chi1**2)) > 1:
        raise Exception('chi1 out of range.')
    if np.sqrt(np.sum(chi2**2)) > 1:
        raise Exception('chi2 out of range.')
    if len(chi1) != 3:
        raise Exception('chi1 must have size 3.')
    if len(chi2) != 3:
        raise Exception('chi2 must have size 3.')

    # use M=10 and distance=1 Mpc, but will scale these out before outputting h
    M = 10      # dimless mass
    distance = 1.0e6 * lal.PC_SI

    approxTag = lalsim.GetApproximantFromString(approximant)
    MT = M*lal.MTSUN_SI
    f_low = Momega0/np.pi/MT
    f_ref = f_low

    # component masses of the binary
    m1_kg = M * lal.MSUN_SI * q / (1. + q)
    m2_kg = M * lal.MSUN_SI / (1. + q)

    # tidal parameters if given
    if lambda1 is not None or lambda2 is not None:
        dictParams = lal.CreateDict()
        lalsim.SimInspiralWaveformParamsInsertTidalLambda1(dictParams, lambda1)
        lalsim.SimInspiralWaveformParamsInsertTidalLambda2(dictParams, lambda2)
    else:
        dictParams = None

    hp, hc = lalsim.SimInspiralChooseTDWaveform(
        m1_kg, m2_kg, chi1[0], chi1[1], chi1[2], chi2[0], chi2[1], chi2[2],
        distance, inclination, phi_ref,
        longAscNodes, eccentricity, meanPerAno,
        deltaTOverM*MT, f_low, f_ref, dictParams, approxTag)

    h = np.array(hp.data.data - 1.j*hc.data.data)
    t = deltaTOverM * np.arange(len(h))  # dimensionless time

    return t, h*distance/MT/lal.C_SI
