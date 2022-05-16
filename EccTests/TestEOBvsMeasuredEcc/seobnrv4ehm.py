"""Python wrapper for SEOBNRv4EHM model by Toni."""
import lal
import lalsimulation
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import os
import h5py
import pycbc.types.timeseries as pt
from pycbc.waveform.utils import taper_timeseries
from pycbc.filter import match
import pycbc.psd.analytical as psda
from scipy.optimize import minimize_scalar
import glob
from tqdm import tqdm

LALMTSUNSI = 4.925491025543575903411922162094833998e-6
# (* Subscript[M, \[CircleDot]][s] = G/c^3Subscript[M, \[CircleDot]][kg]
# ~ 4.93 10^-6s is the geometrized solar mass in seconds.
# [LAL_MTSUN_SI] defined in  lal/src/std/LALConstants.h *)
LALCSI = 299792458.0
# (* LAL_C_SI 299792458e0 /**< Speed of light in vacuo, m s^-1 *)
LALPCSI = 3.085677581491367e16
# (* LAL_PC_SI 3.085677581491367278913937957796471611e16 /**< Parsec, m * *)


def dlm(ell, m, theta):
    """Wigner d function.

    parameters
    ----------
    ell: lvalue
    m: mvalue
    theta: theta angle, e. g in GW, inclination angle iota

    Returns:
    value of d^{ell m}(theta)
    """
    kmin = max(0, m-2)
    kmax = min(ell+m, ell-2)
    d = 0
    for k in range(kmin, kmax+1):
        numerator = np.sqrt(
            float(np.math.factorial(ell+m)
                  * np.math.factorial(ell-m)
                  * np.math.factorial(ell+2)
                  * np.math.factorial(ell-2)))
        denominator = (
            np.math.factorial(k-m+2)
            * np.math.factorial(ell+m-k)
            * np.math.factorial(ell-k-2))
        d += (((-1)**k/np.math.factorial(k))
              * (numerator/denominator)
              * (np.cos(theta/2))**(2*ell+m-2*k-2)
              * (np.sin(theta/2))**(2*k-m+2))
    return d


def ylm(ell, m, theta, phi):
    """Spin weighted spherical harmonics.

    parameters:
    -----------
    ell: lvalue
    m: mvalue
    theta: theta angle, e. g in GW, inclination angle iota
    phi: phi angle, e. g. in GW, orbital phase

    Returns:
    --------
    ylm_s(theta, phi)
    """
    return (np.sqrt((2*ell + 1)/(4*np.pi))
            * dlm(ell, m, theta)
            * np.exp(1j * m * phi))


def AmpPhysicaltoNRTD(ampphysical, M, dMpc):
    return ampphysical*dMpc*1000000*LALPCSI/(LALCSI*(M*LALMTSUNSI))


def compute_freqInterp(time, hlm):
    philm = np.unwrap(np.angle(hlm))

    intrp = InterpolatedUnivariateSpline(time, philm)
    omegalm = intrp.derivative()(time)

    return omegalm


def SectotimeM(seconds, M):
    return seconds/(M*LALMTSUNSI)


def get_sphtseries(q: float, chi1: float, chi2: float,
                   eccentricity: float,
                   eccentric_anomaly: float,
                   f_min: float, M_fed: float, delta_t: float,
                   EccIC: int,
                   dMpc: float,
                   approx: str):

    # Some internal settings of the model
    HypPphi0, HypR0, HypE0 = [0., 0, 0]
    EccFphiPNorder = 99
    EccFrPNorder = 99
    EccWaveformPNorder = 16
    EccBeta = 0.09
    Ecct0 = 100

    EccPNFactorizedForm = EccNQCWaveform = EccPNRRForm = EccPNWfForm = 1
    EccAvNQCWaveform = 1
    EcctAppend = 40
    # EccIC = 0
    # eccentric_anomaly = 0.0

    m1 = q/(1+q)*M_fed
    m2 = 1/(1+q)*M_fed
    dist = dMpc*(1e6*lal.PC_SI)
    SpinAlignedVersion = 41
    nqcCoeffsInput = lal.CreateREAL8Vector(50)

    sphtseries, dyn, dynHi = lalsimulation.SimIMRSpinAlignedEOBModesEcc_opt(
        delta_t,
        m1*lal.MSUN_SI,
        m2*lal.MSUN_SI,
        f_min,
        dist,
        chi1,
        chi2,
        eccentricity,
        eccentric_anomaly,
        SpinAlignedVersion, 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.,
        nqcCoeffsInput, 0,
        EccFphiPNorder, EccFrPNorder, EccWaveformPNorder,
        EccPNFactorizedForm, EccBeta, Ecct0, EccNQCWaveform,
        EccPNRRForm, EccPNWfForm, EccAvNQCWaveform,
        EcctAppend, EccIC, HypPphi0, HypR0, HypE0)
    return sphtseries, dyn, dynHi


def get_modes(
        q=1,
        chi1=0,
        chi2=0,
        eccentricity=1e-5,
        eccentric_anomaly=0,
        f_min=20,
        M_fed=50,
        delta_t=1/2048,
        EccIC=0,
        dMpc=500,
        approx="SEOBNRv4EHM_opt",
        physical_units=True,
        save=True):
    """Get the hlm modes."""
    mode_dir = "./seobmode_data"
    if not os.path.exists(mode_dir):
        os.mkdir(mode_dir)
    mode_file = (f"{mode_dir}/q_{q}_ecc_{eccentricity}_ano_"
                 f"{eccentric_anomaly}_f_min_{f_min}_chi1_{chi1}_chi2_{chi2}_"
                 f"M_fed_{M_fed}_srate_{int(1/delta_t)}_EccIC_{EccIC}_"
                 f"dMpc_{dMpc}_approx_{approx}_phys_units_{physical_units}.h5")
    hlm = {}
    mode_list = [[2, 2],
                 [3, 3],
                 [4, 4],
                 [5, 5]]
    if os.path.exists(mode_file):
        mode_data = h5py.File(mode_file, "r")
        time_array = mode_data["time"][:]
        for ell, m in mode_list:
            hlm[ell, m] = mode_data[f"strain/({ell, m})"][:]
        mode_data.close()
    else:
        sphtseries, dyn, dynHi = get_sphtseries(
            q, chi1, chi2,
            eccentricity,
            eccentric_anomaly,
            f_min,
            M_fed,
            delta_t,
            EccIC,
            dMpc,
            approx)

        # 55 mode
        modeL = sphtseries.l
        modeM = sphtseries.m
        h55 = sphtseries.mode.data.data  # This is h_55
        hlm[modeL, modeM] = h55  # AmpPhysicaltoNRTD(h55, M_fed, dMpc)
        # hlm[modeL, -modeM] = ((-1)**modeL) * np.conjugate(h55)

        # 44 mode
        modeL = sphtseries.next.l
        modeM = sphtseries.next.m
        h44 = sphtseries.next.mode.data.data  # This is h_44
        hlm[modeL, modeM] = h44  # AmpPhysicaltoNRTD(h44, M_fed, dMpc)
        # hlm[modeL, -modeM] = ((-1)**modeL) * np.conjugate(h44)

        # 21 mode
        # modeL = sphtseries.next.next.l
        # modeM = sphtseries.next.next.m
        # h21 = sphtseries.next.next.mode.data.data  # This is h_21
        # hlm[modeL, modeM] = AmpPhysicaltoNRTD(h21, M_fed, dMpc)
        # hlm[modeL, -modeM] = ((-1)**modeL) * np.conjugate(h21)

        # 33 mode
        modeL = sphtseries.next.next.next.l
        modeM = sphtseries.next.next.next.m
        h33 = sphtseries.next.next.next.mode.data.data  # This is h_33
        hlm[modeL, modeM] = h33  # AmpPhysicaltoNRTD(h33, M_fed, dMpc)
        # hlm[modeL, -modeM] = ((-1)**modeL) * np.conjugate(h33)

        # 22 mode
        modeL = sphtseries.next.next.next.next.l
        modeM = sphtseries.next.next.next.next.m
        h22 = sphtseries.next.next.next.next.mode.data.data  # This is h_22
        hlm[modeL, modeM] = h22  # AmpPhysicaltoNRTD(h22, M_fed, dMpc)
        # hlm[modeL, -modeM] = ((-1)**modeL) * np.conjugate(h22)

        time_array = np.arange(0, len(h22)*delta_t, delta_t)
        if not physical_units:
            time_array = SectotimeM(time_array, M_fed)
            for key in hlm.keys():
                hlm[key] = AmpPhysicaltoNRTD(hlm[key], M_fed, dMpc)

        if save:
            mode_data = h5py.File(mode_file, "w")
            mode_data["time"] = time_array
            for ell, m in mode_list:
                mode_data[f"strain/({ell, m})"] = hlm[ell, m]
            mode_data.close()

    return time_array, hlm
