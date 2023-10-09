"""Custom exception classes for gw_eccentricity."""


class InsufficientExtrema(Exception):
    """Exception raised when the number of extrema is not enough.

    While measuring eccentricity, one common failure that may occur is due to
    an insufficient number of extrema. Applying gw_eccentricity to a large
    number of waveforms, for example, when reconstructing the PE posterior by
    measuring eccentricity at the samples, one may need to loop over all the
    samples. In such cases, one may want to avoid failures that are due to an
    insufficient number of extrema. Having a specific exception class helps in
    such scenarios instead of using a generic exception.

    Parameters
    ----------
    extrema_type : str
        Type of extrema. Can be "pericenter" or "apocenter".
    num_extrema : int
        Number of extrema.
    additional_message : str
        Any additional message to append to the exception message.
        Default is None, which adds no additional message.
    """

    def __init__(self, extrema_type, num_extrema, additional_message=None):
        """Init for InsufficientExtrema Class."""
        self.extrema_type = extrema_type
        self.num_extrema = num_extrema
        self.additional_message = additional_message
        self.message = (f"Number of {self.extrema_type} is {self.num_extrema}."
                        f" Number of {self.extrema_type} is not sufficient "
                        f"to build frequency interpolant.")
        if self.additional_message is not None:
            self.message += "\n" + self.additional_message
        super().__init__(self.message)


class NotInAllowedInputRange(Exception):
    """Exception raised when the reference point is outside the allowed range.

    Due to the nature of the eccentricity definition, one can measure the
    eccentricity only within an allowed range of time/frequency. If the failure
    during eccentricity measurement is due to an input time/frequency that lies
    outside the allowed range, this exception helps in identifying that.

    Parameters
    ----------
    reference_point_type : str
        Type of reference point. Can be "time" or "frequency".
    lower : float
        Minimum allowed value, i.e., the lower boundary of the allowed range.
    upper : float
        Maximum allowed value, i.e., the upper boundary of the allowed range.
    additional_message : str
        Any additional message to append to the exception message.
        Default is None, which adds no additional message.
    """

    def __init__(self, reference_point_type, lower, upper,
                 additional_message=None):
        """Init for NotInAllowedRange Class."""
        self.reference_point_type = reference_point_type
        self.lower = lower
        self.upper = upper
        self.additional_message = additional_message
        self.message = (f"{self.reference_point_type} is outside "
                        "the allowed "
                        f"range [{self.lower}, {self.upper}].")
        if self.additional_message is not None:
            self.message += "\n" + self.additional_message
        super().__init__(self.message)
