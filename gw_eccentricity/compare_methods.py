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
        Default is None.
    tmax:
        If not None, errors are computed only for times earlier than tmax.
        Default is None.

    Returns:
    -------
    t:
        Times where errors are computed.
    ecc_errors:
        Absolute errors in eccentricity measured by method 1 (gwecc_obj1) and
        method 2 (gwecc_obj2).
    mean_ano_errors:
        Absolute errors in mean anomaly measured by method 1 (gwecc_obj1) and
        method 2 (gwecc_obj2).
    """
    # Check that the gwecc objects were created using same tref_in
    np.testing.assert_allclose(gwecc_obj1.tref_in, gwecc_obj2.tref_in)
    # Get the bounds for times within which both methods work
    tMinCommon = max(gwecc_obj1.t_min, gwecc_obj2.t_min)
    tMaxCommon = min(gwecc_obj1.t_max, gwecc_obj2.t_max)
    # Get common tref_out
    common_tref_out = gwecc_obj1.tref_out[
        np.logical_and(gwecc_obj1.tref_out >= tMinCommon,
                       gwecc_obj1.tref_out < tMaxCommon)]
    # Truncate common tref_out if tmin/tmax is provided
    if tmin is not None:
        if all(common_tref_out < tmin):
            raise Exception(f"No common time found later than {tmin}")
        common_tref_out = common_tref_out[common_tref_out >= tmin]
    if tmax is not None:
        if all(common_tref_out > tmax):
            raise Exception(f"No common time found earlier than {tmax}")
        common_tref_out = common_tref_out[common_tref_out <= tmax]

    # Get indices for the common tref_out
    common_idx1 = np.logical_and(gwecc_obj1.tref_out >= common_tref_out[0],
                                 gwecc_obj1.tref_out <= common_tref_out[-1])
    common_idx2 = np.logical_and(gwecc_obj2.tref_out >= common_tref_out[0],
                                 gwecc_obj2.tref_out <= common_tref_out[-1])
    # Check that common tref out is the same.
    np.testing.assert_allclose(gwecc_obj1.tref_out[common_idx1],
                               gwecc_obj2.tref_out[common_idx2])
    # Compute errors in eccentricity
    ecc_errors = np.abs(
        gwecc_obj1.ecc_ref[common_idx1] - gwecc_obj2.ecc_ref[common_idx2])
    # Compute errors in mean anomaly
    # We need to unwrap the mean anomaly since zero and
    # 2pi should be treated as the same and hence zero errors.
    mean_ano_errors = np.abs(
        np.unwrap(gwecc_obj1.mean_ano_ref[common_idx1])
        - np.unwrap(gwecc_obj2.mean_ano_ref[common_idx2]))
    return gwecc_obj1.tref_out[common_idx1], ecc_errors, mean_ano_errors
