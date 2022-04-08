"""Useful functions for the project."""
import numpy as np
import lal
import lalsimulation as lalsim


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
    MT = M * lal.MTSUN_SI
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


def check_kwargs_and_set_defaults(user_kwargs=None,
                                  default_kwargs=None,
                                  name="user given kwargs"):
    """Sanity check user given dicionary of kwargs and set default values.

    parameters:
    user_kwargs: Dictionary of kwargs by user
    default_kwargs: Dictionary of default kwargs
    name: string to represnt the dictionary
    """
    for kw in user_kwargs.keys():
        if kw not in default_kwargs:
            raise ValueError(f"Invalid key {kw} in {name}."
                             " Should be one of "
                             f"{default_kwargs.keys()}")

    for kw in default_kwargs.keys():
        if kw not in user_kwargs:
            user_kwargs[kw] = default_kwargs[kw]
