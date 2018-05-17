from . import utils
from .base import BaseModel
import numpy as np


class ThermalTime(BaseModel):
    """Thermal Time Model

    The classic growing degree day model using a fixed temperature
    threshold above which forcing accumulates.

    Event happens on :math:`DOY` when the following is met:

    .. math::
        \sum_{t=t_{1}}^{DOY}R_{f}(T_{i})\geq F^{*}

    where:

    .. math::
        R_{f}(T_{i}) = max(T_{i}-threshold, 0)

    Parameters:
        t1 : int
            | :math:`t_{1}` - The DOY which forcing accumulating beings
            | default : (-67,298)

        T : int
            | :math:`T` - The threshold above which forcing accumulates
            | default : (-25,25)

        F : int, > 0
            | :math:`F^{*}` - The total forcing units required
            | default : (0,1000)

    """

    def __init__(self, parameters={}):
        BaseModel.__init__(self)
        self.all_required_parameters = {'t1': (-67, 298), 'T': (-25, 25), 'F': (0, 1000)}
        self._organize_parameters(parameters)
        self._required_data = {'predictor_columns': ['site_id', 'year', 'doy', 'temperature'],
                               'predictors': ['temperature', 'doy_series']}

    def _apply_model(self, temperature, doy_series, t1, T, F):
        # Temperature threshold
        temperature[temperature < T] = 0

        # Only accumulate forcing after t1
        temperature[doy_series < t1] = 0

        accumulated_gdd = utils.transforms.forcing_accumulator(temperature)

        return utils.transforms.doy_estimator(forcing=accumulated_gdd,
                                              doy_series=doy_series,
                                              threshold=F)

class M1(BaseModel):
    """The Thermal Time Model with a daylength correction.

    Event happens on :math:`DOY` when the following is met:

    .. math::

        \sum_{t=t_{1}}^{DOY}R_{f}(T_{i}) \geq (L_{i}/24)^kF^*

    where:

    .. math::
        R_{f}(T_{i}) = max(T_{i}-threshold, 0)

    This model requires a daylength column in the predictors in
    addition to daily mean temperature.

    Parameters:
        t1 : int
            | :math:`t_{1}` - The DOY which forcing accumulating beings
            | default : (-67,298)

        T : int
            | :math:`T` - The threshold above which forcing accumulates
            | default : (-25,25)

        F : int, > 0
            | :math:`F^{*}` - The total forcing units required
            | default : (0,1000)

        k : int, > 0
            | :math:`k^{*}` - Daylength coefficient
            | default : (0,50)

    Notes:
        Blümel, K., & Chmielewski, F. M. (2012). Shortcomings of classical phenological
        forcing models and a way to overcome them.
        Agricultural and Forest Meteorology, 164, 10–19.
        http://doi.org/10.1016/j.agrformet.2012.05.001

    """

    def __init__(self, parameters={}):
        BaseModel.__init__(self)
        self.all_required_parameters = {'t1': (-67, 298), 'T': (-25, 25),
                                        'F': (0, 1000), 'k': (0, 50)}
        self._organize_parameters(parameters)
        self._required_data = {'predictor_columns': ['site_id', 'year', 'doy', 'temperature', 'daylength'],
                               'predictors': ['temperature', 'doy_series', 'daylength']}

    def _organize_predictors(self, predictors, observations, for_prediction):
        """Convert data to internal structure used by M1 model
        """
        # daylength for each observation. 1d array with length n, which should match
        # axis 0 of temperature_array
        obs_with_daylength = observations.merge(predictors, on=['site_id', 'year'], how='left')
        daylength_array = obs_with_daylength.daylength.values

        if for_prediction:
            temperature_array, doy_series = utils.misc.temperature_only_data_prep(observations=observations,
                                                                                  predictors=predictors,
                                                                                  for_prediction=for_prediction)
            return {'temperature': temperature_array,
                    'daylength': daylength_array,
                    'doy_series': doy_series}
        else:
            cleaned_observations, temperature_array, doy_series = utils.misc.temperature_only_data_prep(observations,
                                                                                                        predictors,
                                                                                                        for_prediction=for_prediction)
            self.fitting_predictors = {'temperature': temperature_array,
                                       'daylength': daylength_array,
                                       'doy_series': doy_series}

            self.obs_fitting = cleaned_observations

    def _validate_formatted_predictors(self, predictors):
        pass

    def _apply_model(self, temperature, doy_series, daylength, t1, T, F, k):
        # Temperature threshold
        temperature[temperature < T] = 0

        # Only accumulate forcing after t1
        temperature[doy_series < t1] = 0

        accumulated_gdd = utils.transforms.forcing_accumulator(temperature)

        daylength_adjustment = (daylength / 24) ** k
        # Make it the same shape as gdd for easy adjustment
        num_days = len(doy_series)
        daylength_adjustment = np.tile(np.expand_dims(daylength_adjustment, 1), num_days)

        return utils.transforms.doy_estimator(forcing=accumulated_gdd,
                                              doy_series=doy_series,
                                              threshold=F)

class FallCooling(BaseModel):
    """Fall senesence model

    A model for fall senesence. Essential a Thermal Time model,
    but instead of accumulating warming above a base temperature,
    it accumulates cooling below a max temperature. 

    Event happens on :math:`DOY` when the following is met:

    .. math::
        \sum_{t=t_{1}}^{DOY}R_{f}(T_{i})\geq F^{*}

    where:

    .. math::
        R_{f}(T_{i}) = max(threshold-T_{i}, 0)
    
    This is a simplified version of the model in Delpierre et al. 2009.
    The full version also has a photoperiod compoenent. 

    Parameters:
        t1 : int
            | :math:`t_{1}` - The DOY which forcing accumulating beings
            | default : (182,365)

        T : int
            | :math:`T` - The threshold below which cooling accumulates
            | default : (-25,25)

        F : int, > 0
            | :math:`F^{*}` - The total cooling units required
            | default : (0,1000)

    Notes:
        Delpierre, N., Dufrêne, E., Soudani, K., Ulrich, E., Cecchini, S., Boé,
        J., & François, C. (2009). Modelling interannual and spatial
        variability of leaf senescence for three deciduous tree species in
        France. Agricultural and Forest Meteorology, 149(6–7), 938–948.
        http://doi.org/10.1016/j.agrformet.2008.11.014

    """

    def __init__(self, parameters={}):
        BaseModel.__init__(self)
        self.all_required_parameters = {'t1': (182, 365), 'T': (-25, 25), 'F': (0, 1000)}
        self._organize_parameters(parameters)
        self._required_data = {'predictor_columns': ['site_id', 'year', 'doy', 'temperature'],
                               'predictors': ['temperature', 'doy_series']}

    def _apply_model(self, temperature, doy_series, t1, T, F):
        # Temperature threshold
        temperature[temperature > T] = 0

        # Only accumulate forcing after t1
        temperature[doy_series < t1] = 0

        accumulated_gdd = utils.transforms.forcing_accumulator(temperature)

        return utils.transforms.doy_estimator(forcing=accumulated_gdd,
                                              doy_series=doy_series,
                                              threshold=F)