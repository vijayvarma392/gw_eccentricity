import gw_eccentricity
from gw_eccentricity import load_data
from gw_eccentricity import measure_eccentricity
from gw_eccentricity.utils import time_deriv_4thOrder
import numpy as np
import copy


def compare_eccentricity_measurement(dataDict1, dataDict2, method):
    """Compare the measured eccentricity and mean anomaly.

    Eccentricity measurement can take different combination of waveform data.
    Two input dictionaries with different but allowed combination of data from
    the same waveform should result in the same measured eccentricity and mean
    anomaly.

    Parameters
    ----------
    dataDict1 : dict
        Dictionary with allowed combination of waveform data.
    dataDict2 : dict
        Dictionary with allowed combination of waveform data.
    method: str
        Method to use for eccentricity measurement.
    """
    # Check that both the dataDict has same time arrays
    if any(dataDict1["t"] != dataDict2["t"]):
        raise Exception("The dataDicts must have the same time array.")
    result1 = measure_eccentricity(
        tref_in=dataDict1["t"],
        method=method,
        dataDict=dataDict1)
    gwecc_object1 = result1["gwecc_object"]
    eccentricity1 = gwecc_object1.eccentricity
    mean_anomaly1 = gwecc_object1.mean_anomaly

    result2 = measure_eccentricity(
        tref_in=dataDict1["t"],
        method=method,
        dataDict=dataDict2)
    gwecc_object2 = result2["gwecc_object"]
    eccentricity2 = gwecc_object2.eccentricity
    mean_anomaly2 = gwecc_object2.mean_anomaly

    # Compare the eccentricity and mean anomaly
    np.testing.assert_allclose(eccentricity1,
                               eccentricity2)
    np.testing.assert_allclose(mean_anomaly1,
                               mean_anomaly2)


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
    # check that the time array is uniform. This is required to obtain omega by
    # taking derivative of phase applying time_deriv_4thorder.
    t_diff = np.diff(dataDict["t"])
    if not np.allclose(t_diff, t_diff[0]):
        raise Exception("Input time array must have uniform time steps.\n"
                        f"Time steps are {t_diff}")
    for k in dataDict["hlm"]:
        phaselm[k] = -np.unwrap(np.angle(dataDict["hlm"][k]))
        amplm[k] = np.abs(dataDict["hlm"][k])
        omegalm[k] = time_deriv_4thOrder(phaselm[k], t_diff[0])

    # get amplm_zeroecc, phaselm_zeroecc, omegalm_zeroecc
    phaselm_zeroecc = {}
    amplm_zeroecc = {}
    omegalm_zeroecc = {}
    t_zeroecc_diff = np.diff(dataDict["t_zeroecc"])
    if not np.allclose(t_zeroecc_diff, t_zeroecc_diff[0]):
        raise Exception("Input time array must have uniform time steps.\n"
                        f"Time steps are {t_zeroecc_diff}")
    for k in dataDict["hlm_zeroecc"]:
        phaselm_zeroecc[k] = -np.unwrap(np.angle(dataDict["hlm_zeroecc"][k]))
        amplm_zeroecc[k] = np.abs(dataDict["hlm_zeroecc"][k])
        omegalm_zeroecc[k] = time_deriv_4thOrder(phaselm_zeroecc[k],
                                                 t_zeroecc_diff[0])

    # List of all available methods
    available_methods = gw_eccentricity.get_available_methods()

    # create data dict with hlm
    dataDict1_w_hlm = {
        "t": dataDict["t"],
        "hlm": dataDict["hlm"]}
    # create data dict with hlm and omegalm
    dataDict2_w_hlm_and_omega = copy.deepcopy(dataDict1_w_hlm)
    dataDict2_w_hlm_and_omega.update({"omegalm": omegalm})
    # create data dict with amplm and phaselm
    dataDict2_w_amplm_phaselm = {
        "t": dataDict["t"],
        "amplm": amplm,
        "phaselm": phaselm}
    # create data dict with amplm, phaselm and omegalm
    dataDict2_w_amplm_phaselm_and_omegalm = copy.deepcopy(
        dataDict2_w_amplm_phaselm)
    dataDict2_w_amplm_phaselm_and_omegalm.update({"omegalm": omegalm})

    # We loop over different data dict and compare the eccentricity measurement
    # It checks that the eccentricity measurement is the same when we provide
    # any of the following combinations:
    # - only hlm
    # - hlm and omegalm
    # - only amplm and phaselm
    # - amplm, phaselm and omegalm
    for dataDict2 in [dataDict2_w_hlm_and_omega,
                      dataDict2_w_amplm_phaselm,
                      dataDict2_w_amplm_phaselm_and_omegalm]:
        # Loop over the different methods
        for method in available_methods:
            if "Residual" in method:
                # To avoid exception for residual method since the data does
                # not contain zeroecc data
                continue
            # use data_w_hlm as dataDict1
            compare_eccentricity_measurement(dataDict1=dataDict1_w_hlm,
                                             dataDict2=dataDict2,
                                             method=method)
    # Now we repeat the same as above but adding hlm_zeroecc to both the data
    # dict.
    # Add hlm_zeroecc to dataDict1 first so that dataDict1 now contains
    # hlm and hlm_zeroecc.
    dataDict1_w_hlm_zeroecc = copy.deepcopy(dataDict1_w_hlm)
    dataDict1_w_hlm_zeroecc.update(
        {"t_zeroecc": dataDict["t_zeroecc"],
         "hlm_zeroecc": dataDict["hlm_zeroecc"]})
    # The following checks that the eccentricity measurement is the same
    # when we provide:
    # - hlm and hlm_zeroecc with or without omegalm_zeroecc
    # - hlm, omegalm and hlm_zeroecc with or without omegalm_zeroecc
    # - amplm, phaselm and hlm_zeroecc with or without omegalm_zeroecc
    # - amplm, phaselm, omegalm and hlm_zeroecc with or without omegalm_zeroecc
    for dataDict2 in [dataDict2_w_hlm_and_omega,
                      dataDict2_w_amplm_phaselm,
                      dataDict2_w_amplm_phaselm_and_omegalm]:
        dataDict2_w_hlm_zeroecc = copy.deepcopy(dataDict2)
        dataDict2_w_hlm_zeroecc.update(
            {"t_zeroecc": dataDict["t_zeroecc"],
             "hlm_zeroecc": dataDict["hlm_zeroecc"]})
        # add omegalm_zeroecc also
        dataDict2_w_hlm_zeroecc_and_omegalm_zeroecc = copy.deepcopy(
            dataDict2_w_hlm_zeroecc)
        dataDict2_w_hlm_zeroecc_and_omegalm_zeroecc.update(
            {"omegalm_zeroecc": omegalm_zeroecc})
        # Loop over the different methods
        for method in available_methods:
            compare_eccentricity_measurement(
                dataDict1=dataDict1_w_hlm_zeroecc,
                dataDict2=dataDict2_w_hlm_zeroecc,
                method=method)
            compare_eccentricity_measurement(
                dataDict1=dataDict1_w_hlm_zeroecc,
                dataDict2=dataDict2_w_hlm_zeroecc_and_omegalm_zeroecc,
                method=method)
    # This time instead of adding hlm_zeroecc, we add amplm_zeroecc and
    # phaselm_zeroecc to dataDict2, dataDict1 is the same as above.
    # The following checks that the eccentricity measurement is the same
    # when we provide:
    # - hlm and hlm_zeroecc with or without omegalm_zeroecc
    # - hlm, omegalm, amplm_zeroecc and phaselm_zeroecc with or without omegalm_zeroecc
    # - amplm, phaselm, amplm_zeroecc and phaselm_zeroecc with or without omegalm_zeroecc
    # - amplm, phaselm, omegalm, amplm_zeroecc and phaselm_zeroecc with or without omegalm_zeroecc
    for dataDict2 in [dataDict2_w_hlm_and_omega,
                      dataDict2_w_amplm_phaselm,
                      dataDict2_w_amplm_phaselm_and_omegalm]:
        dataDict2_w_amplm_and_phaselm_zeroecc = copy.deepcopy(dataDict2)
        dataDict2_w_amplm_and_phaselm_zeroecc.update(
            {"t_zeroecc": dataDict["t_zeroecc"],
             "amplm_zeroecc": amplm_zeroecc,
             "phaselm_zeroecc": phaselm_zeroecc})
        # add omegalm_zeroecc also
        dataDict2_w_amplm_and_phaselm_and_omegalm_zeroecc = copy.deepcopy(
            dataDict2_w_amplm_and_phaselm_zeroecc)
        dataDict2_w_amplm_and_phaselm_and_omegalm_zeroecc.update(
            {"omegalm_zeroecc": omegalm_zeroecc})
        # Loop over the different methods
        for method in available_methods:
            compare_eccentricity_measurement(
                dataDict1=dataDict1_w_hlm_zeroecc,
                dataDict2=dataDict2_w_amplm_and_phaselm_zeroecc,
                method=method)
            compare_eccentricity_measurement(
                dataDict1=dataDict1_w_hlm_zeroecc,
                dataDict2=dataDict2_w_amplm_and_phaselm_and_omegalm_zeroecc,
                method=method)
