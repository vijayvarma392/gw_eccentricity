"""Utility to load waveform data from lvcnr files or LAL."""
import numpy as np
import gwtools
from .utils import generate_waveform
from .utils import get_peak_via_quadratic_fit
import h5py
import lal
import lalsimulation as lalsim


def load_waveform(catalog="LAL", **kwargs):
    """Load waveform from lvcnr file or LAL.

    parameters:
    ----------
    catalog:
          Waveform type. could be one of 'LAL', 'LVCNR', EOB

    kwargs:
         Kwargs to be passed to the waveform loading functions.
    """
    if catalog == "LAL":
        return load_LAL_waveform(**kwargs)
    elif catalog == "LVCNR":
        if kwargs["filepath"] is None:
            raise Exception("Must provide file path to NR waveform")
        return load_lvcnr_waveform(**kwargs)
    elif catalog == "EOB":
        if kwargs["filepath"] is None:
            raise Exception("Must provide file path to EOB waveform")
        return load_h22_from_EOBfile(**kwargs)
    else:
        raise Exception(f"Unknown catalog {catalog}")


def load_LAL_waveform(**kwargs):
    """FIXME add some documentation."""
    # FIXME: Generalize this
    if 'deltaTOverM' not in kwargs:
        kwargs['deltaTOverM'] = 0.1

    if 'approximant' in kwargs:
        # FIXME, this assumes single mode models, talk to Vijay about
        # how to handle other models.
        dataDict = load_LAL_waveform_using_hack(
                kwargs['approximant'],
                kwargs['q'],
                kwargs['chi1'],
                kwargs['chi2'],
                kwargs['ecc'],
                kwargs['mean_ano'],
                kwargs['Momega0'],
                kwargs['deltaTOverM'])
    else:
        raise Exception("HELP! Only know how to do LAL waveforms for now.")

    if ('include_zero_ecc' in kwargs) and kwargs['include_zero_ecc']:
        # Keep all other params fixed but set ecc=0.
        zero_ecc_kwargs = kwargs.copy()
        # FIXME: Stupic EccentricTD only works for finite ecc
        if kwargs["approximant"] == "EccentricTD":
            zero_ecc_kwargs['ecc'] = 1e-5
        else:
            zero_ecc_kwargs['ecc'] = 0
        zero_ecc_kwargs['include_zero_ecc'] = False   # to avoid infinite loops
        dataDict_zero_ecc = load_waveform(**zero_ecc_kwargs)
        t_zeroecc = dataDict_zero_ecc['t']
        hlm_zeroecc = dataDict_zero_ecc['hlm']
        dataDict.update({'t_zeroecc': t_zeroecc,
                         'hlm_zeroecc': hlm_zeroecc})
    return dataDict


def load_LAL_waveform_using_hack(approximant, q, chi1, chi2, ecc, mean_ano,
                                 Momega0, deltaTOverM):
    """Load LAL waveforms."""
    # Many LAL models don't return the modes. So, to get h22 we evaluate the
    # strain at (incl, phi)=(0,0) and divide by Ylm(0,0).  NOTE: This only
    # works if the only mode is the (2,2) mode.
    phi_ref = 0
    inclination = 0

    # h = hp -1j * hc
    t, h = generate_waveform(approximant, q, chi1, chi2,
                             deltaTOverM, Momega0, eccentricity=ecc,
                             phi_ref=phi_ref, inclination=inclination)

    Ylm = gwtools.harmonics.sYlm(-2, 2, 2, inclination, phi_ref)
    mode_dict = {(2, 2): h/Ylm}
    # Make t = 0 at the merger. This would help when getting
    # residual amplitude by subtracting quasi-circular counterpart
    t = t - get_peak_via_quadratic_fit(t, np.abs(h))[0]

    dataDict = {"t": t, "hlm": mode_dict}
    return dataDict


def time_to_physical(M):
    """Factor to convert time from dimensionless units to SI units.

    parameters
    ----------
    M: mass of system in the units of solar mass

    Returns
    -------
    converting factor
    """
    return M * lal.MTSUN_SI


def amp_to_physical(M, D):
    """Factor to rescale amp from dimensionless units to SI units.

    parameters
    ----------
    M: mass of the system in units of solar mass
    D: Luminosity distance in units of megaparsecs

    Returns
    -------
    Scaling factor
    """
    return lal.G_SI * M * lal.MSUN_SI / (lal.C_SI**2 * D * 1e6 * lal.PC_SI)


def load_lvcnr_waveform(filepath, modeList=[[2, 2]], M=50, dt=1/4096,
                        dist_mpc=1, f_low=0, dimensionless=True):
    """Load modes from lvcnr files.

    parameters:
    ----------
    filepath:
        Path to lvcnr file.

    modeList:
        modes to inlcude if mode is not "all".

    If f_low = 0, uses the entire NR data. The actual f_low will be
    returned.
    """
    NRh5File = h5py.File(filepath, 'r')

    # set mode for NR data
    params_NR = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertNumRelData(params_NR, filepath)

    # Metadata parameters masses:
    m1 = NRh5File.attrs['mass1']
    m2 = NRh5File.attrs['mass2']
    m1SI = m1 * M / (m1 + m2) * lal.MSUN_SI
    m2SI = m2 * M / (m1 + m2) * lal.MSUN_SI

    distance = dist_mpc * 1.0e6 * lal.PC_SI
    f_ref = f_low
    spins = lalsim.SimInspiralNRWaveformGetSpinsFromHDF5File(f_ref, M,
                                                             filepath)
    s1x = spins[0]
    s1y = spins[1]
    s1z = spins[2]
    s2x = spins[3]
    s2y = spins[4]
    s2z = spins[5]

    # If f_low == 0, update it to the start frequency so that the surrogate
    # gets the right start frequency
    if f_low == 0:
        f_low = NRh5File.attrs['f_lower_at_1MSUN'] / M
    f_ref = f_low
    f_low = f_ref

    # Generating the NR modes
    values_mode_array = lalsim.SimInspiralWaveformParamsLookupModeArray(
        params_NR)
    _, modes = lalsim.SimInspiralNRWaveformGetHlms(
        dt,
        m1SI,
        m2SI,
        distance,
        f_low,
        f_ref,
        s1x,
        s1y,
        s1z,
        s2x,
        s2y,
        s2z,
        filepath,
        values_mode_array)
    mode = modes
    factor_to_convert_time_to_dimensionless = (
        1 / time_to_physical(M) if dimensionless else 1)
    factor_to_convert_amp_to_dimensionless = (
        (1 / amp_to_physical(M, dist_mpc)) if dimensionless else 1)
    modes_dict = {}
    while 1 > 0:
        try:
            l, m = mode.l, mode.m
            read_mode = mode.mode.data.data
            modes_dict[(l, m)] = (read_mode
                                  * factor_to_convert_amp_to_dimensionless)
            mode = mode.next
        except AttributeError:
            break

    t = np.arange(len(modes_dict[(l, m)])) * dt
    t = t * factor_to_convert_time_to_dimensionless
    t = t - get_peak_via_quadratic_fit(t, np.abs(modes_dict[(2, 2)]))[0]

    q = m1SI/m2SI
    try:
        eccentricity = float(NRh5File.attrs["eccentricity"])
    except ValueError:
        eccentricity = None

    NRh5File.close()

    return_dict = {"t": t,
                   "hlm": modes_dict,
                   "q": q,
                   "ecc": eccentricity,
                   "spins": [s1x, s1y, s1z, s2x, s2y, s2z],
                   "flow": f_low,
                   "f_ref": f_ref}
    return return_dict


def load_h22_from_EOBfile(EOB_file):
    """Load data from EOB files."""
    fp = h5py.File(EOB_file, "r")
    t_ecc = fp['data/t'][:]
    amp22_ecc = fp['data/hCoOrb/Amp_l2m2'][:]
    phi22_ecc = fp['data/hCoOrb/phi_l2m2'][:]

    t_nonecc = fp['data/t'][:]
    amp22_nonecc = fp['nonecc_data/hCoOrb/Amp_l2m2'][:]
    phi22_nonecc = fp['nonecc_data/hCoOrb/phi_l2m2'][:]

    fp.close()
    dataDict = {"t": t_ecc, "hlm": amp22_ecc * np.exp(1j * phi22_ecc),
                "t_zeroecc": t_nonecc,
                "hlm_zeroecc": amp22_nonecc * np.exp(1j * phi22_nonecc)}
    return dataDict
