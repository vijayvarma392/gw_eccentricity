"""Compare different eccDefinition methods."""

import numpy as np


def compute_errors_between_methods(gwecc_obj1,
                                   gwecc_obj2,
                                   tmin=None,
                                   tmax=None):
    """Compute errors in eccentricity and mean anomaly from two methods.

    This function computes the errors (difference) in the measured value of
    eccentricity and mean anomaly using two different methods. Since both
    methods might be not able to measure eccentricity and mean anomaly at
    the same range of times, this function will return the errors in the region
    of common times.

    Parameters:
    -----------
    gwecc_obj1:
        gwecc_object using method 1 as returned by
        gw_eccentricity.measure_eccentricity function with
        "return_gwecc_object" set to True.
    gwecc_obj2:
        gwecc_object using method 2 as returned by
        gw_eccentricity.measure_eccentricity function with
        "return_gwecc_object" set to True.
    tmin:
        If not None, errors are computed only for times later than tmin.
        If None, errors are computed for times later than
        max(gwecc_obj1.tmin, gwecc_obj2.tmin).
        Default is None.
    tmax:
        If not None, errors are computed only for times earlier than tmax.
        If None, errors are computed for times earlier than
        min(gwecc_obj1.tmax, gwecc_obj2.tmax).
        Default is None.

    Returns:
    -------
    t:
        Times where errors are computed. The minimum/maximum t is choosen
        such that the times in t are common to both methods. That is,
        tmin = max(gwecc_obj1.tmin, gwecc_obj2.tmin) and
        tmax = min(gwecc_obj1.tmax, gwecc_obj2.tmax).
        If in addition, tmin/tmax is not None, then t is further truncated
        such that t lies within tmin/tmax.
    ecc_errors:
        Absolute errors in eccentricity measured by method 1 (gwecc_obj1) and
        method 2 (gwecc_obj2).
    mean_ano_errors:
        Absolute errors in mean anomaly measured by method 1 (gwecc_obj1) and
        method 2 (gwecc_obj2).
    ecc_ref_1_common:
        Eccentricities at t measured using method 1.
    ecc_ref_2_common:
        Eccentricities at t measured using method 2.
    mean_ano_ref_1_common:
        Mean anomalies at t measured using method 1.
    mean_ano_ref_2_common:
        Mean anomalies at t measured using method 2.
    """
    # Check that the gwecc objects were created using same tref_in
    np.testing.assert_allclose(gwecc_obj1.tref_in, gwecc_obj2.tref_in)
    # Get the bounds for times within which both methods work
    tMinCommon = max(gwecc_obj1.tmin, gwecc_obj2.tmin)
    tMaxCommon = min(gwecc_obj1.tmax, gwecc_obj2.tmax)

    # Reset tMinCommon/tMaxCommon if tmin/tmax is provided
    if tmin is not None:
        if tmin > tMaxCommon:
            raise Exception(f"No common time found later than {tmin}")
        tMinCommon = max(tMinCommon, tmin)
    if tmax is not None:
        if tmax < tMinCommon:
            raise Exception(f"No common time found earlier than {tmax}")
        tMaxCommon = min(tMaxCommon, tmax)

    # Get indices for the common tref_out
    common_idx1 = np.logical_and(gwecc_obj1.tref_out >= tMinCommon,
                                 gwecc_obj1.tref_out < tMaxCommon)
    common_idx2 = np.logical_and(gwecc_obj2.tref_out >= tMinCommon,
                                 gwecc_obj2.tref_out < tMaxCommon)
    tref_out_1_common = gwecc_obj1.tref_out[common_idx1]
    tref_out_2_common = gwecc_obj2.tref_out[common_idx2]
    # Check that common tref out is the same.
    np.testing.assert_allclose(tref_out_1_common, tref_out_2_common)
    # Compute errors in eccentricity at common tref_out
    ecc_ref_1_common = gwecc_obj1.eccentricity[common_idx1]
    ecc_ref_2_common = gwecc_obj2.eccentricity[common_idx2]
    ecc_errors = np.abs(ecc_ref_1_common - ecc_ref_2_common)
    # Compute errors in mean anomaly at common tref_out
    mean_ano_ref_1_common = gwecc_obj1.mean_anomaly[common_idx1]
    mean_ano_ref_2_common = gwecc_obj2.mean_anomaly[common_idx2]
    # We need to unwrap the mean anomaly since zero and
    # 2pi should be treated as the same and hence should give zero errors.
    mean_ano_errors = np.abs(
        np.unwrap(mean_ano_ref_1_common)
        - np.unwrap(mean_ano_ref_2_common))
    return tref_out_1_common, ecc_errors, mean_ano_errors, ecc_ref_1_common,\
        ecc_ref_2_common, mean_ano_ref_1_common, mean_ano_ref_2_common
