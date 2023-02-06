import gw_eccentricity
from gw_eccentricity import load_data
from gw_eccentricity import measure_eccentricity
import numpy as np
from lal import MTSUN_SI


def test_mks_vs_dimless_units():
    """ Tests that the measure_eccentricity interface is working for both
    MKS and dimensionless units.

    We pass dataDict created using EccentricTD to the ecc measurement method in
    dimensionless unit and physical unit (MKS) and compare the measured
    eccentricity and mean anomaly for
    - a fixed time
    - an array of times
    - a fixed frequency
    - an array of frequencies.

    We expect the measured eccenrtcity and mean anomaly for each of the above
    cases to be the same whether the data is in dimensionless units or physical
    units.
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
    # load_waveform returns dimensionless dataDict
    dataDict = load_data.load_waveform(**lal_kwargs)
    # get dataDict in MKS units.
    lal_kwargs.update({"physicalUnits": True,
                       "M": 10,
                       "D": 1})
    dataDictMKS = load_data.load_waveform(**lal_kwargs)

    # List of all available methods
    available_methods = gw_eccentricity.get_available_methods()
    for method in available_methods:
        # Try evaluating at a single dimless time t = -12000
        idx = np.argmin(np.abs(dataDict["t"] - (-12000)))
        gwecc_dict = measure_eccentricity(
            tref_in=dataDict["t"][idx],
            method=method,
            dataDict=dataDict)
        tref_out = gwecc_dict["tref_out"]
        ecc_ref = gwecc_dict["eccentricity"]
        meanano_ref = gwecc_dict["mean_anomaly"]
        gwecc_object = gwecc_dict["gwecc_object"]
        # Try evaluating at a single MKS time
        gwecc_dict_MKS = measure_eccentricity(
            tref_in=dataDictMKS["t"][idx],
            method=method,
            dataDict=dataDictMKS)
        tref_out_MKS = gwecc_dict_MKS["tref_out"]
        ecc_ref_MKS = gwecc_dict_MKS["eccentricity"]
        meanano_ref_MKS = gwecc_dict_MKS["mean_anomaly"]
        gwecc_object_MKS = gwecc_dict_MKS["gwecc_object"]
        # Check that the tref_out is the same as rescaled (to dimless) tref_out_MKS
        sec_to_dimless = 1/lal_kwargs["M"]/MTSUN_SI
        np.testing.assert_allclose(
            [tref_out],
            [tref_out_MKS * sec_to_dimless],
            err_msg=("tref_out at a single dimensionless and MKS"
                     " time are inconsistent.\n"
                     "x = Dimensionless, y = MKS converted to dimless"))
        # Check if the measured ecc an mean ano are the same from the two units
        np.testing.assert_allclose(
            [ecc_ref],
            [ecc_ref_MKS],
            err_msg=("Eccentricity at a single dimensionless and MKS"
                     " time gives different results.\n"
                     "x = Dimensionless, y = MKS"))
        np.testing.assert_allclose(
            [meanano_ref],
            [meanano_ref_MKS],
            err_msg=("Mean anomaly at a single dimensionless and MKS"
                     " time gives different results.\n"
                     "x = Dimensionless, y = MKS"))

        # Try evaluating at an array of dimless times
        idx_start = np.argmin(np.abs(dataDict["t"] - (-12000)))
        idx_end = np.argmin(np.abs(dataDict["t"] - (-5000)))
        gwecc_dict = measure_eccentricity(
            tref_in=dataDict["t"][idx_start: idx_end],
            method=method,
            dataDict=dataDict)
        tref_out = gwecc_dict["tref_out"]
        ecc_ref = gwecc_dict["eccentricity"]
        meanano_ref = gwecc_dict["mean_anomaly"]
        # Try evaluating at an array of MKS times
        gwecc_dict_MKS = measure_eccentricity(
            tref_in=dataDictMKS["t"][idx_start: idx_end],
            method=method,
            dataDict=dataDictMKS)
        tref_out_MKS = gwecc_dict_MKS["tref_out"]
        ecc_ref_MKS = gwecc_dict_MKS["eccentricity"]
        meanano_ref_MKS = gwecc_dict_MKS["mean_anomaly"]
        # check that the tref_out times are consistent
        np.testing.assert_allclose(
            [tref_out],
            [tref_out_MKS * sec_to_dimless],
            err_msg=("tref_out array for dimensionless and MKS"
                     " tref_in are inconsistent.\n"
                     "x = Dimensionless, y = MKS converted to dimless"))
        # Check if the measured ecc an mean ano are the same from the two units
        np.testing.assert_allclose(
            ecc_ref,
            ecc_ref_MKS,
            err_msg=("Eccentricity at dimensionless and MKS array of"
                     " times are different\n."
                     "x = Dimensionless, y = MKS"))
        # using unwrapped mean anomaly since 0 and 2pi should be treated as
        # the same
        np.testing.assert_allclose(
            np.unwrap(meanano_ref),
            np.unwrap(meanano_ref_MKS),
            err_msg=("Mean anomaly at dimensionless and MKS array of"
                     " times are different.\n"
                     "x = Dimensionless, y = MKS"))

        # Try evaluating at single dimensionless frequency
        fref_in = gwecc_object.compute_orbit_averaged_omega22_at_extrema(
            dataDict["t"][idx]) / (2 * np.pi)
        gwecc_dict = measure_eccentricity(
            fref_in=fref_in,
            method=method,
            dataDict=dataDict)
        fref_out = gwecc_dict["fref_out"]
        ecc_ref = gwecc_dict["eccentricity"]
        meanano_ref = gwecc_dict["mean_anomaly"]
        # Try evaluating at single MKS frequency
        fref_in = gwecc_object_MKS.compute_orbit_averaged_omega22_at_extrema(
            dataDictMKS["t"][idx]) / (2 * np.pi)
        gwecc_dict_MKS = measure_eccentricity(
            fref_in=fref_in,
            method=method,
            dataDict=dataDictMKS)
        fref_out_MKS = gwecc_dict_MKS["fref_out"]
        ecc_ref_MKS = gwecc_dict_MKS["eccentricity"]
        meanano_ref_MKS = gwecc_dict_MKS["mean_anomaly"]
        # Check the fref_out frequencies are consistent
        np.testing.assert_allclose(
            [fref_out],
            [fref_out_MKS / sec_to_dimless],
            err_msg=("fref_out for a single dimensionless and MKS"
                     " fref_in are inconsistent.\n"
                     "x = Dimensionless, y = MKS converted to dimless"))
        # Check if the measured ecc an mean ano are the same from the two units
        np.testing.assert_allclose(
            [ecc_ref],
            [ecc_ref_MKS],
            err_msg=("Eccentricity at a single dimensionless and MKS"
                     " frequency gives different results.\n"
                     "x = Dimensionless, y = MKS"))
        np.testing.assert_allclose(
            [meanano_ref],
            [meanano_ref_MKS],
            err_msg=("Mean anomaly at a single dimensionless and MKS"
                     " frequency gives different results.\n"
                     "x = Dimensionless, y = MKS"))

        # Try evaluating at an array of dimensionless frequencies
        fref_in = gwecc_object.compute_orbit_averaged_omega22_at_extrema(
            dataDict["t"][idx: idx+500]) / (2 * np.pi)
        gwecc_dict = measure_eccentricity(
            fref_in=fref_in,
            method=method,
            dataDict=dataDict)
        fref_out = gwecc_dict["fref_out"]
        ecc_ref = gwecc_dict["eccentricity"]
        meanano_ref = gwecc_dict["mean_anomaly"]
        # Try evaluating at an array of MKS frequencies
        fref_in = gwecc_object_MKS.compute_orbit_averaged_omega22_at_extrema(
            dataDictMKS["t"][idx: idx+500]) / (2 * np.pi)
        gwecc_dict_MKS = measure_eccentricity(
            fref_in=fref_in,
            method=method,
            dataDict=dataDictMKS)
        fref_out_MKS = gwecc_dict_MKS["fref_out"]
        ecc_ref_MKS = gwecc_dict_MKS["eccentricity"]
        meanano_ref_MKS = gwecc_dict_MKS["mean_anomaly"]
        # Check fref_out
        np.testing.assert_allclose(
            [fref_out],
            [fref_out_MKS / sec_to_dimless],
            err_msg=("fref_out for an array of dimensionless and MKS"
                     " fref_in are inconsistent.\n"
                     "x = Dimensionless, y = MKS converted to dimless"))
        # Check if the measured ecc an mean ano are the same from the two units
        np.testing.assert_allclose(
            ecc_ref,
            ecc_ref_MKS,
            err_msg=("Eccentricity at dimensionless and MKS array of"
                     " frequencies are different.\n"
                     "x = Dimensionless, y = MKS"))
        np.testing.assert_allclose(
            np.unwrap(meanano_ref),
            np.unwrap(meanano_ref_MKS),
            err_msg=("Mean anomaly at dimensionless and MKS array of"
                     " frequencies are different.\n"
                     "x = Dimensionless, y = MKS"))
