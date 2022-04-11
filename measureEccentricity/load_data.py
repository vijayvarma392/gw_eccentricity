"""Utility to load waveform data from lvcnr files or LAL."""
import numpy as np
import gwtools
from .utils import generate_waveform
from .utils import get_peak_via_quadratic_fit
import h5py
import lal
import lalsimulation as lalsim
from pycbc.pnutils import mtotal_eta_to_mass1_mass2


def load_waveform(catalog="LAL",
                  lvcnr_kwargs=None,
                  lal_kwargs=None,
                  eob_kwargs=None):
    """Load waveform from lvcnr file or LAL.

    parameters:
    ----------
    catalog:
          Waveform type. could be one of 'LAL', 'LVCNR', EOB

    lvcnr_kwargs:
         kwargs to be passed to load lvcnr files.

    lal_kwargs:
        kwargs to generate LAL waveform.

    eob_kwargs:
        kwargs to load EOB waveforms
    """
    if catalog == "LAL":
        return load_LAL_waveform(**lal_kwargs)
    elif catalog == "LVCNR":
        if lvcnr_kwargs["path_to_file"] is None:
            raise Exception("Must provide file path to NR waveform")
        return load_from_lvcnr_file(**lvcnr_kwargs)
    elif catalog == "EOB":
        if eob_kwargs["path_to_file"] is None:
            raise Exception("Must provide file path to EOB waveform")
        return load_h22_from_EOBfile(**eob_kwargs)
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
        dataDict_zero_ecc = load_waveform(lal_kwargs=zero_ecc_kwargs)
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


def get_lal_mode_dictionary(mode_array):
    """
    Get LALDict with all specified modes.

    Parameters
    ----------
    mode_array: list of modes, eg [[2,2], [3,3]]

    Returns
    -------
    waveform_dictionary: LALDict with all modes included

    """
    waveform_dictionary = lal.CreateDict()
    mode_array_lal = lalsim.SimInspiralCreateModeArray()
    for mode in mode_array:
        lalsim.SimInspiralModeArrayActivateMode(
            mode_array_lal, mode[0], mode[1])
    lalsim.SimInspiralWaveformParamsInsertModeArray(
        waveform_dictionary, mode_array_lal)

    return waveform_dictionary


def load_from_lvcnr_file(path_to_file,
                         Mtot=50,
                         distance=1,
                         srate=4096,
                         modeList=[[2, 2]],
                         f_low=None,
                         dimensionless=True):
    """Get individual modes from LVCNR format file.

    Parameters
    ==========
    path_to_file: string
        Path to LVCNR file
    Mtot: float
        Total mass (in units of MSUN) to scale the waveform to
    distance: float
        Luminosity Distance (in units of MPc) to scale the waveform to
    srate: int
        Sampling rate for the waveform
    modeList: list
        List of modes to use.
        (Default: [[2, 2]])
    f_low: float
        Value of the low frequency to start waveform generation
        Uses value given from the LVCNR file if `None` is provided
        (Default: None)

    Returns
    =======
    mass_ratio: float
        Mass ratio derived from the LVCNR file
    spins_args: list
        List of spins derived from the LVCNR file
    eccentricity: float
        Eccentricty derived from the LVCNR file.
        Returns `None` is eccentricity is not a number.
    f_low: float
        Low Frequency derived either from the file, or provided
        in the call
    f_ref: float
        Reference Frequency derived from the file
    modes: dict of pycbc TimeSeries objects
        dict containing all the read in modes
    """
    with h5py.File(path_to_file) as h5file:
        waveform_dict = get_lal_mode_dictionary(modeList)

        f_low_in_file = h5file.attrs["f_lower_at_1MSUN"] / Mtot
        f_ref = f_low_in_file

        if f_low is None:
            f_low = f_low_in_file

        if h5file.attrs["Format"] < 3:
            spin_args = lalsim.SimInspiralNRWaveformGetSpinsFromHDF5File(
                -1, Mtot, path_to_file
            )
        else:
            spin_args = lalsim.SimInspiralNRWaveformGetSpinsFromHDF5File(
                f_low, Mtot, path_to_file
            )

        mass_args = list(mtotal_eta_to_mass1_mass2(Mtot * lal.MSUN_SI,
                                                   h5file.attrs["eta"]))

        try:
            eccentricity = float(h5file.attrs["eccentricity"])
        except ValueError:
            eccentricity = None

    values_mode_array = lalsim.SimInspiralWaveformParamsLookupModeArray(
        waveform_dict)
    _, modes = lalsim.SimInspiralNRWaveformGetHlms(
        1 / srate,
        *mass_args,
        distance * 1e6 * lal.PC_SI,
        f_low,
        f_ref,
        *spin_args,
        path_to_file,
        values_mode_array,
    )
    mode = modes
    factor_to_convert_time_to_dimensionless = (
        1 / time_to_physical(Mtot) if dimensionless else 1)
    factor_to_convert_amp_to_dimensionless = (
        (1 / amp_to_physical(Mtot, distance)) if dimensionless else 1)
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

    t = np.arange(len(modes_dict[(l, m)])) / srate
    t = t * factor_to_convert_time_to_dimensionless
    t = t - get_peak_via_quadratic_fit(t, np.abs(modes_dict[(2, 2)]))[0]

    return_dict = {"t": t,
                   "hlm": modes_dict,
                   "q": mass_args[1] / mass_args[0],
                   "ecc": eccentricity,
                   "spins": spin_args,
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
