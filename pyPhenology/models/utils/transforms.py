import numpy as np
from scipy.special import expit


def mean_temperature(temperature, doy_series, start_doy, end_doy):
    """Mean temperature of a single time period.
    ie. mean spring temperature.

    Parameters
    ----------
    temperature : Numpy array
        array of daily temperature values

    doy_series : Numpy array
        1D array as produced by format_data(),
        identifying the doy values in forcing[:,b]

    start_doy : int
        The beginning of the time period

    end_doy : int
        The end of the time period
    """
    if start_doy > end_doy:
        raise RuntimeError('start_doy must be < end_doy')

    spring_days = np.logical_and(doy_series >= start_doy, doy_series <= end_doy)
    return temperature[spring_days].mean(axis=0)


def triangle_response(temperature, t_min, t_opt, t_max):
    """Triangle function

    Used to simulate and optimal temperature between a low and high temperature.
    """
    outside_triangle = np.logical_or(temperature <= t_min, temperature >= t_max)
    left_side = np.logical_and(temperature > t_min, temperature <= t_opt)
    right_side = np.logical_and(temperature > t_opt, temperature < t_max)

    temperature[left_side] -= t_min
    temperature[left_side] /= t_opt - t_min

    temperature[right_side] -= t_max
    temperature[right_side] /= t_opt - t_max

    temperature[outside_triangle] = 0

    return temperature


def sigmoid2(temperature, b, c):
    """The two parameter sigmoid function from Chuine 2000

    The full equation is f(x) = 1 / (1 + exp(b*(temp -c)))

    The scipy.special.expit function = 1/(1+(-x))

    Using float32 increases speed significantly over 
    the default float64

    Parameters
    ----------
    temperature : Numpy array
        array of daily temperature values

    b : int
        Sigmoid fitting parameter

    c : int
        Sigmoid fitting parameter

    Returns
    -------
    temperature : Numpy array
        array of daily forcings derived from function
    """
    return expit(-(b * (temperature.astype(np.float32) - c)))


def sigmoid3(temperature, a, b, c):
    """The three parameter sigmoid function from Chuine 2000

    The full equation is f(x) = 1 / (1 + exp(a*(temp-c)**2 +  b*(temp -c)))

    The scipy.special.expit function = 1/(1+(-x))

    Using float32 increases speed significantly over 
    the default float64

    Parameters
    ----------
    temperature : Numpy array
        (obs,doy) array of daily temperature values

    a : int
        Sigmoid fitting parameter

    b : int
        Sigmoid fitting parameter

    b : int
        Sigmoid fitting parameter

    Returns
    -------
    temperature : Numpy array
        array of daily forcings derived from function
    """
    return expit(-(a * ((temperature.astype(np.float32) - c)**2) + b * (temperature.astype(np.float32) - c)))


def daylength(doy, latitude):
    """Calculates daylength in hours

    From https://github.com/khufkens/phenor
    """
    assert isinstance(doy, np.ndarray), 'doy should be np array'
    assert isinstance(latitude, np.ndarray), 'latitude should be np array'
    assert doy.shape == latitude.shape, 'latitude and doy should be equal lengths'
    assert len(doy.shape) == 1, 'doy should be 1 dimensional'
    doy = doy.copy()
    latitude = latitude.copy()

    # negative doy values used in pyPhenology should be converted back to
    # positive for daylength calculation
    doy[doy < 1] += 365

    # set constants
    latitude = (np.pi / 180) * latitude

    # Correct for winter solstice
    doy += 11

    # earths ecliptic
    j = np.pi / 182.625
    axis = (np.pi / 180) * 23.439

    m = 1 - np.tan(latitude) * np.tan(axis * np.cos(j * doy))

    # sun never appears or disappears
    m = np.maximum(m, 0)
    m = np.minimum(m, 2)

    # Exposed fraction of the sun's circle
    b = np.arccos(1 - m) / np.pi

    # Daylength (lat,day)
    b *= 24

    return b


def forcing_accumulator(temperature):
    """ The accumulated forcing for each observation
    and doy in the (obs, doy) array.
    """
    return temperature.cumsum(axis=0)


def doy_estimator(forcing, doy_series, threshold, non_prediction=999):
    """ Find the doy that some forcing threshold is met for a large
    number of sites.

    Parameters
    ----------
    forcing : Numpy array
        Either a 2d or 3d array holding timeseries of
        daily mean temperature value of different replicates.
        The 0 axis is always the time axis. Axis 1 in a 2d array
        is the number of replicates. Axis 1 and 2 in a 3d array
        are the spatial replicates (ie lat, lon)
        values are the accumulated forcing for 
        each replicate,doy.

    doy_series : Numpy array
        1D array as produced by format_data(),
        identifying the doy values in forcing[0]

    threshold : float | int
        Threshold that must be met in forcing

    non_prediction : int
        Value to return if the threshold value is not
        met. A large value should be used during fitting
        to ensure unrealistic parameters are not chosen.

    Returns
    -------
    doy_final : Numpy array
        1D array of length obs with the doy values which
        first meet the threshold
    """
    # If threshold is not met for a particular row, ensure that a large doy
    # gets returned so it produces a large error
    non_prediction_buffer = np.expand_dims(np.zeros_like(forcing[0]), axis=0)
    non_prediction_buffer[:] = 10e5
    forcing = np.concatenate((forcing, non_prediction_buffer), axis=0)
    doy_series = np.append(doy_series, non_prediction)

    # The index of the doy for each element where F was met
    doy_with_threshold_met = np.argmax(forcing >= threshold, axis=0)

    doy_final = np.take(doy_series, doy_with_threshold_met)

    return doy_final
