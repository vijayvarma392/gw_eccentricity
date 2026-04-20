"""Core postprocessing functions for measuring eccentricity from waveform modes."""
from __future__ import annotations
from dataclasses import dataclass
import logging
import pandas as pd
from ..gw_eccentricity import measure_eccentricity

logger = logging.getLogger(__name__)

@dataclass
class PostProcessResult:
    sample_index: int
    status: str
    egw: float | None
    lgw: float | None
    error_message: str | None = None

@dataclass
class PostProcessResults:
    results: list[PostProcessResult]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the list of PostProcessResult to a pandas DataFrame."""
        return pd.DataFrame([result.__dict__ for result in self.results])

    def success_only(self) -> list[PostProcessResult]:
        """Filter successful results."""
        return [r for r in self.results if r.status == "success"]

    def get_summary(self) -> dict:
        """Get summary statistics."""
        successful = self.success_only()
        total = len(self.results)
        return {
            'total_samples': total,
            'success_percentage': (len(successful) / total) * 100,
            'egw': [r.egw for r in successful],
            'lgw': [r.lgw for r in successful]
        }

@dataclass
class FrefBoundsResult:
    sample_index: int
    status: str
    fref_min: float | None
    fref_max: float | None
    error_message: str | None = None


@dataclass
class FrefBoundsResults:
    results: list[FrefBoundsResult]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the list of FrefBoundsResult to a pandas DataFrame."""
        return pd.DataFrame([result.__dict__ for result in self.results])
    
    def success_only(self) -> list[FrefBoundsResult]:
        """Filter successful results."""
        return [r for r in self.results if r.status == "success"]
    
    def get_summary(self) -> dict:
        """Get summary statistics."""
        successful = self.success_only()
        total = len(self.results)
        return {
            'total_samples': total,
            'success_percentage': (len(successful) / total) * 100,
            'fref_min': [r.fref_min for r in successful],
            'fref_max': [r.fref_max for r in successful]
        }

    def get_minmax_fref(self) -> tuple[float, float] | None:
        """Get the min and max fref across all successful samples.
        
        This provides the common range of fref values where eccentricity can be
        measured for all the successful samples.
        """
        summary = self.get_summary()
        if summary['success_percentage'] == 0:
            raise Exception("No successful samples to determine fref bounds.")
        return max(summary['fref_min']), min(summary['fref_max'])


def filter_posterior_columns(
        posterior: pd.DataFrame,
        parameter_columns: list[str]
) -> pd.DataFrame:
    """Filter posterior DataFrame to only include columns used by
    ``data_dict_generator``.

    Parameters
    ----------
    posterior : pd.DataFrame
        Full posterior DataFrame.
    parameter_columns : list[str]
        Column names required by the data_dict_generator.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with only the requested columns.
    """
    missing = [k for k in parameter_columns if k not in posterior.columns]
    if missing:
        raise ValueError(f"Columns not found in posterior: {missing}")
    return posterior[parameter_columns].copy()


def get_data_dict(
        params: dict,
        data_dict_generator: callable,
        extra_kwargs: dict | None = None
    ) -> dict:
    """Get data_dict for given params in the posterior.

    Parameters
    ----------
    params : dict
        Dictionary containing the parameters for the sample.

    data_dict_generator : function
        data_dict is generated using function call as below::

            data_dict = data_dict_generator(params, extra_kwargs)

    extra_kwargs : dict, optional
        Extra kwargs passed to ``data_dict_generator``.

    Returns
    -------
    data_dict : dict
        Dictionary of waveform modes data compatible with
        ``gw_eccentricity.measure_eccentricity``.
    """
    if extra_kwargs is None:
        extra_kwargs = {}
    data_dict = data_dict_generator(
        params, extra_kwargs)
    if not isinstance(data_dict, dict):
        raise TypeError(
            f"The data_dict generator `{data_dict_generator}` should "
            f"return a dict and not a {type(data_dict)}")
    return data_dict


def get_fref_bounds_for_sample(
        sample_index: int,
        params: dict,
        data_dict_generator: callable,
        data_dict_generator_extra_kwargs: dict | None = None,
        method: str = "Amplitude",
        gw_eccentricity_kwargs: dict | None = None
        ) -> FrefBoundsResult:
    """Get the min and max allowed fref for a given sample.

    Parameters
    ----------
    sample_index : int
        Index of the sample in the posterior.
    params : dict
        Dictionary containing the parameters for the sample.
    data_dict_generator : function
        Function to generate the data dictionary for the sample.
    data_dict_generator_extra_kwargs : dict, optional
        Extra kwargs passed to ``data_dict_generator``.
    method : str, default="Amplitude"
        Method to use in ``gw_eccentricity.measure_eccentricity``.
    gw_eccentricity_kwargs : dict, optional
        Extra kwargs passed to ``gw_eccentricity.measure_eccentricity``.

    Returns
    -------
    FrefBoundsResult
        with keys ``sample_index``, ``status``, ``fref_min``, ``fref_max``,
        and on failure ``error_message``.
    """
    if gw_eccentricity_kwargs is None:
        gw_eccentricity_kwargs = {}
    try:
        data_dict = get_data_dict(params, data_dict_generator, data_dict_generator_extra_kwargs)
        res = measure_eccentricity(
            dataDict=data_dict,
            tref_in=data_dict["t"], # pass the full time array to get the fref bounds for the entire waveform
            method=method,
            **gw_eccentricity_kwargs)
        gw_obj = res["gwecc_object"]
        fref_bounds = gw_obj.get_fref_bounds()
        return FrefBoundsResult(
            sample_index=sample_index,
            status="success",
            fref_min=fref_bounds[0],
            fref_max=fref_bounds[1]
        )
    except Exception as e:
        logger.warning(f" Sample {params} failed to get fref bounds: {e}")
        return FrefBoundsResult(
            sample_index=sample_index,
            status="fail",
            fref_min=None,
            fref_max=None,
            error_message=str(e))

def postprocess_sample(
        sample_index: int,
        params: dict,
        tref: float | None,
        fref: float | None,
        data_dict_generator: callable,
        data_dict_generator_extra_kwargs: dict | None = None,
        method: str = "Amplitude",
        gw_eccentricity_kwargs: dict | None = None) -> PostProcessResult:
    """Measure eccentricity and mean anomaly from waveform modes for a sample.

    A wrapper around ``gw_eccentricity.measure_eccentricity`` to measure
    eccentricity from the waveform modes for a sample with given ``params``.

    Eccentricity can be measured either at a reference time ``tref`` or
    a reference frequency ``fref``. It can not accept both at the same time.

    Allowed reference times where eccentricity can be measured is determined
    by the bounds [tref_min, tref_max] where:

    - tref_min is the max(time of first pericenter, time of first apocenter)
    - tref_max is the min(time of last pericenter, time of last apocenter)

    Allowed reference frequencies where eccentricity can be measured is determined
    by the bounds [fref_min, fref_max] which are the min and max values of the
    omega_gw_average, respectively (See eccDefinition.get_omega_gw_average).
    One can get these bounds by running ``get_fref_bounds_for_sample``.

    Parameters
    ----------
    sample_index : int
        Index of the sample in the posterior.
    params : dict
        Dictionary containing the parameters for the sample.
    tref : float | None
        Reference time where eccentricity is to be measured.
    fref : float | None
        Reference frequency where eccentricity is to be measured.
    data_dict_generator : function
        data_dict is generated using function call as below::
            data_dict = data_dict_generator(params, data_dict_generator_extra_kwargs)
    data_dict_generator_extra_kwargs : dict, optional
        Extra kwargs passed to ``data_dict_generator``.
    method : str, default="Amplitude"
        Method to use in ``gw_eccentricity.measure_eccentricity``.
    gw_eccentricity_kwargs : dict, optional
        Extra kwargs passed to ``gw_eccentricity.measure_eccentricity``.

    Returns
    -------
    PostProcessResult
        with keys: ``status``, ``egw``, ``lgw``, and on failure
        ``error_message``.
    """
    if not ((tref is None) ^ (fref is None)):
        raise ValueError("Provide exactly one of tref or fref, but not both.")
    try:
        data_dict = get_data_dict(
            params,
            data_dict_generator,
            data_dict_generator_extra_kwargs)
        res = measure_eccentricity(
            dataDict=data_dict,
            tref_in=tref,
            fref_in=fref,
            method=method,
            **(gw_eccentricity_kwargs or {}))
        return PostProcessResult(
            sample_index=sample_index,
            status="success",
            egw=res["eccentricity"],
            lgw=res["mean_anomaly"]
        )
    except Exception as e:
        logger.warning(f" Sample {params} failed: {e}")
        return PostProcessResult(
            sample_index=sample_index,
            status="fail",
            egw=None,
            lgw=None,
            error_message=str(e))
