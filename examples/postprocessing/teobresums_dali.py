"""Example code for generating TEOBResumS-DALI waveforms to be used by
gw_eccentricity when postprocessing bilby posterior samples.

We use teob_data_dict_generator as the data_dict_generator for postprocessing
the bilby posterior samples in example notebook postprocess_for_bilby.ipynb and
in the script postprocess.sh

Thanks to Danilo Chiaramello for helping with the implementation of
the backward evolution in teob_data_dict_generator.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/arif/teobresums_reviewed/Python/")
import EOBRun_module as EOB


def teob_data_dict_generator(params, kwargs=None):
    """Generate a data dictionary.

    Generate teobresums-dali waveform modes to be used by
    gw_eccentricity.
    """
    if kwargs is None:
        kwargs = {}
    teob_params = {
            'M'                  : params['mass_1'] + params['mass_2'],
            'q'                  : params['mass_1'] / params['mass_2'],
            'chi1z'              : params['spin_1z'],
            'chi2z'              : params['spin_2z'],
            'distance'           : params['luminosity_distance'],
            'initial_frequency'  : params["minimum_frequency"],
            'use_geometric_units': "no",
            'use_mode_lm'        : [0, 1, 4, 8],
            'arg_out'            : "yes",        # Output dynamics and hlm in addition to h+, hx
            'ecc'                : params['eccentricity'],
            'anomaly'            : params['mean_per_ano'],
            'time_shift_TD'      : "no",         # This is so the model doesn't shift the time axis to set the amplitude peak at t = 0
            'interp_uniform_grid': "yes",
        }
    
    # forward integration
    u_f, _, _, hlm_f, _ = EOB.EOBRunPy(teob_params)
    A22 = hlm_f['1'][0]
    p22 = hlm_f['1'][1]
    t   = u_f

    # backward integration
    teob_params['backwards'] = kwargs.get('backwards', "no")
    if teob_params['backwards'] == "yes":
        # This is in physical units if not using geometric; maximum time for
        # the backward evolution.
        # It's important to set it explicitly because the default value is very
        # large (unlike the forward evolution, where the merger would normally
        # be expected to terminate the evolution)
        # which may not be necessary.
        teob_params['ode_tmax']  = kwargs.get('ode_tmax', 1.0)

        u_b, _, _, hlm_b, _ = EOB.EOBRunPy(teob_params)

        # flip the backward waveform and join them
        A22_b = hlm_b['1'][0][::-1]
        p22_b = np.unwrap(hlm_b['1'][1][::-1])
        A22   = np.concatenate((A22_b, A22[1:]))
        p22   = np.concatenate((p22_b, p22[1:]))
        t     = t[1:] + u_b[-1]
        t     = np.concatenate((u_b, t))

    return {"t": t,
            "amplm": {(2,2): A22},
            "phaselm": {(2,2): p22}}
