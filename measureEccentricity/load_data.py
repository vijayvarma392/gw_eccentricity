import numpy as np
import gwtools
from .utils import generate_waveform
from .utils import get_peak_via_quadratic_fit

def load_waveform(**kwargs):
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
        zero_ecc_kwargs['ecc'] = 1e-5
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
    t = t - get_peak_via_quadratic_fit(t, np.abs(h))[0]

    dataDict = {"t": t, "hlm": mode_dict}
    return dataDict
