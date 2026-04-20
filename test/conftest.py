"""Shared test helpers and fixtures for the gw_eccentricity test suite."""
import numpy as np
from pyseobnr.generate_waveform import generate_modes_opt


def get_seob_datadict(q=5.0, chi1z=0.4, chi2z=0.3,
                      omega_start=0.02, ecc=0.1, mean_ano=1.7,
                      include_zero_ecc=False):
    """Return a dataDict built from a SEOBNRv5EHM waveform.

    Parameters
    ----------
    q            : mass ratio
    chi1z, chi2z : aligned spin components
    omega_start  : initial orbital frequency in geometric units
    ecc          : initial eccentricity
    mean_ano     : initial mean anomaly
    include_zero_ecc : if True, also add t_zeroecc/hlm_zeroecc keys built
                       from a zero-eccentricity waveform starting slightly
                       earlier (omega_start * 0.9), required by Residual* methods
    """
    t, modes = generate_modes_opt(
        q, chi1z, chi2z, omega_start,
        eccentricity=ecc, rel_anomaly=mean_ano,
        approximant="SEOBNRv5EHM")
    hlm = {tuple(int(x) for x in k.split(",")): v for k, v in modes.items()}
    dataDict = {"t": t, "hlm": hlm}

    if include_zero_ecc:
        t_z, modes_z = generate_modes_opt(
            q, chi1z, chi2z, omega_start * 0.9,
            eccentricity=0, rel_anomaly=0,
            approximant="SEOBNRv5EHM")
        hlm_z = {tuple(int(x) for x in k.split(",")): v for k, v in modes_z.items()}
        dataDict.update({"t_zeroecc": t_z, "hlm_zeroecc": hlm_z})

    return dataDict
