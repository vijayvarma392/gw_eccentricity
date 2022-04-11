import numpy as np
import gwtools
from .utils import generate_waveform
from .utils import get_peak_via_quadratic_fit
import h5py
from scipy.interpolate import InterpolatedUnivariateSpline


def load_waveform(catalog="LAL", file=None, nr_modelist=[[2, 2]], **kwargs):
    """Load waveform from file or LAL

    parameters:
    ----------
    catalog:
          Waveform type. could be one of 'LAL', 'EOB', 'SXS', 'ET'

    file:
         Path to file for waveform to be read from a file. This is required for
         catalog other than 'LAL'

    nr_modelist:
        List of modes to include when loading NR data from 'SXS' or 'ET'

    **kwargs:
        Keywords to generate LAL waveform.
    """
    if catalog == "LAL":
        return load_LAL_waveform(**kwargs)
    elif catalog in ["SXS", "ET"]:
        if file is None:
            raise Exception("Must provide file path to NR waveform")
        return load_from_NR_file(file, nr_modelist, sim_type=catalog)
    elif catalog == "EOB":
        if file is None:
            raise Exception("Must provide file path to EOB waveform")
        return load_h22_from_EOBfile(file)
    else:
        raise Exception(f"Unknown catalog {catalog}")


def load_LAL_waveform(**kwargs):
    """ FIXME add some documentation.
    """
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


def load_from_NR_file(NR_file, ModeList, sim_type='SXS'):
    hlm = {}
    amplm = {}
    phaselm = {}
    omegalm = {}

    fp = h5py.File(NR_file, "r")

    for l, m in ModeList:
        Alm = fp['amp_l'+str(l)+'_m'+str(m)+'/Y'][:]
        tAlm = fp['amp_l'+str(l)+'_m'+str(m)+'/X'][:]

        philm = fp['phase_l'+str(l)+'_m'+str(m)+'/Y'][:]
        tphilm = fp['phase_l'+str(l)+'_m'+str(m)+'/X'][:]

        tNR = fp['NRtimes'][:]
        iAlm = InterpolatedUnivariateSpline(tAlm, Alm)
        iphilm = InterpolatedUnivariateSpline(tphilm, philm)

        amplm[(l, m)] = iAlm(tNR)
        phaselm[(l, m)] = iphilm(tNR)
        omegalm[(l, m)] = iphilm.derivative()(tNR)
        hlm[(l, m)] = amplm[l, m] * np.exp(1.j * phaselm[l, m])

    if sim_type == 'SXS':
        t_omega_orb = fp['Omega-vs-time/X'][:]
        tHorizon = fp['HorizonBTimes'][:]
        omega_orb = fp['Omega-vs-time/Y'][:]
        iOmega_orb = InterpolatedUnivariateSpline(t_omega_orb, omega_orb)

        om_orb = iOmega_orb(tHorizon)
        phi = []

        for time in tHorizon:
            phi.append(iOmega_orb.integral(tHorizon[0], time))
        phase_orb = np.array(phi)
    else:
        phase_orb = None
        om_orb = None
        tHorizon = None
    fp.close()
    dataDict = {"t": tNR, "hlm": hlm, "tHorizon": tHorizon, "amplm": amplm,
                "phaselm": phaselm, "omegalm": omegalm, "om_orb": om_orb,
                "phase_orb": phase_orb}
    return dataDict


def load_h22_from_EOBfile(EOB_file):
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
