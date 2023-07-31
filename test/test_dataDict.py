import gw_eccentricity
from gw_eccentricity import load_data
from gw_eccentricity import measure_eccentricity
from gw_eccentricity.utils import time_deriv_4thOrder
import numpy as np


def test_dataDict():
    """
    Test that the interface works for dataDict with all allowed
    combination of keys in dataDict.
    """
    # Load test waveform
    lal_kwargs = {"approximant": "EccentricTD",
                  "q": 1.0,
                  "chi1": [0.0, 0.0, 0.0],
                  "chi2": [0.0, 0.0, 0.0],
                  "Momega0": 0.01,
                  "ecc": 0.1,
                  "mean_ano": 0,
                  "include_zero_ecc": True}
    dataDict = load_data.load_waveform(**lal_kwargs)

    # get amplm, phaselm, omegalm
    phaselm = {}
    amplm = {}
    omegalm = {}
    dt = dataDict["t"][1] - dataDict["t"][0]
    for k in dataDict["hlm"]:
        phaselm[k] = -np.unwrap(np.angle(dataDict["hlm"][k]))
        amplm[k] = np.abs(dataDict["hlm"][k])
        omegalm[k] = time_deriv_4thOrder(phaselm[k], dt)

    # get amplm_zeroecc, phaselm_zeroecc, omegalm_zeroecc
    phaselm_zeroecc = {}
    amplm_zeroecc = {}
    omegalm_zeroecc = {}
    dt_zeroecc = dataDict["t_zeroecc"][1] - dataDict["t_zeroecc"][0]
    for k in dataDict["hlm_zeroecc"]:
        phaselm_zeroecc[k] = -np.unwrap(np.angle(dataDict["hlm_zeroecc"][k]))
        amplm_zeroecc[k] = np.abs(dataDict["hlm_zeroecc"][k])
        omegalm_zeroecc[k] = time_deriv_4thOrder(phaselm_zeroecc[k],
                                                 dt_zeroecc)

    # List of all available methods
    available_methods = gw_eccentricity.get_available_methods()

    # create data dict with hlm
    data_hlm = {"t": dataDict["t"],
                "hlm": dataDict["hlm"]}
    # create data dict with amplm and phaselm
    data_amplm_phaselm = {
        "t": dataDict["t"],
        "amplm": amplm,
        "phaselm": phaselm}

    for method in available_methods:
        # Using hlm
        if "Residual" in method:
            # add zeroecc hlm
            data_hlm.update({"t_zeroecc": dataDict["t_zeroecc"],
                             "hlm_zeroecc": dataDict["hlm_zeroecc"]})
        return_dict_hlm = measure_eccentricity(
            tref_in=dataDict["t"],
            method=method,
            dataDict=data_hlm)
        gwecc_object_hlm = return_dict_hlm["gwecc_object"]
        eccentricity_w_hlm = gwecc_object_hlm.eccentricity
        mean_anomaly_w_hlm = gwecc_object_hlm.mean_anomaly

        # Using hlm with omegalm also provided
        data_hlm.update({"omegalm": omegalm})
        if "Residual" in method:
            data_hlm.update(
                {"omegalm_zeroecc": omegalm_zeroecc})
        return_dict_hlm_omegalm = measure_eccentricity(
            tref_in=dataDict["t"],
            method=method,
            dataDict=data_hlm)
        gwecc_object_hlm_omegalm = return_dict_hlm_omegalm["gwecc_object"]
        eccentricity_w_hlm_and_omegalm = gwecc_object_hlm_omegalm.eccentricity
        mean_anomaly_w_hlm_and_omegalm = gwecc_object_hlm_omegalm.mean_anomaly

        # using amplm and phaselm
        if "Residual" in method:
            # add zeroecc amplm and phaselm
            data_amplm_phaselm.update(
                {"t_zeroecc": dataDict["t_zeroecc"],
                 "amplm_zeroecc": amplm_zeroecc,
                 "phaselm_zeroecc": phaselm_zeroecc})
        return_dict_amplm_phaselm = measure_eccentricity(
            tref_in=dataDict["t"],
            method=method,
            dataDict=data_hlm)
        gwecc_object_amplm_phaselm = return_dict_amplm_phaselm["gwecc_object"]
        eccentricity_w_amplm_phaselm = gwecc_object_amplm_phaselm.eccentricity
        mean_anomaly_w_amplm_phaselm = gwecc_object_amplm_phaselm.mean_anomaly

        # Using amplm and phaselm with omegalm
        data_amplm_phaselm.update({"omegalm": omegalm})
        if "Residual" in method:
            data_amplm_phaselm.update(
                {"omegalm_zeroecc": omegalm_zeroecc})
        return_dict_amplm_phaselm_and_omegalm = measure_eccentricity(
            tref_in=dataDict["t"],
            method=method,
            dataDict=data_hlm)
        gwecc_object_amplm_phaselm_and_omegalm = return_dict_amplm_phaselm_and_omegalm["gwecc_object"]
        eccentricity_w_amplm_phaselm_and_omegalm = gwecc_object_amplm_phaselm_and_omegalm.eccentricity
        mean_anomaly_w_amplm_phaselm_and_omegalm = gwecc_object_amplm_phaselm_and_omegalm.mean_anomaly

        # Compare eccenrtcity and mean anomaly between hlm and amp/phase
        np.testing.assert_allclose(eccentricity_w_hlm,
                                   eccentricity_w_amplm_phaselm)
        np.testing.assert_allclose(mean_anomaly_w_hlm,
                                   mean_anomaly_w_amplm_phaselm)
        # Compare eccenrtcity and mean anomaly using hlm with and without omega
        np.testing.assert_allclose(eccentricity_w_hlm,
                                   eccentricity_w_hlm_and_omegalm)
        np.testing.assert_allclose(mean_anomaly_w_hlm,
                                   mean_anomaly_w_hlm_and_omegalm)
        # Compare eccenrtcity and mean anomaly using amplm and phaselm with and
        # without omega
        np.testing.assert_allclose(eccentricity_w_amplm_phaselm,
                                   eccentricity_w_amplm_phaselm_and_omegalm)
        np.testing.assert_allclose(mean_anomaly_w_amplm_phaselm,
                                   mean_anomaly_w_amplm_phaselm_and_omegalm)
