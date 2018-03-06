from . import utils
from .base import _base_model
import numpy as np

class ThermalTime(_base_model):
    """Thermal Time Model
    
    The classic growing degree day model using a fixed temperature
    threshold above which forcing accumulates.
    
    Event happens on :math:`DOY` when the following is met:
    
    .. math::
        \sum_{t=t_{1}}^{DOY}R_{f}(T_{i})\geq F^{*}
    
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
        _base_model.__init__(self)
        self.all_required_parameters = {'t1':(-67,298),'T':(-25,25),'F':(0,1000)}
        self._organize_parameters(parameters)
        self._required_data={'predictor_columns':['site_id','year','doy','temperature'],
                             'predictors':['temperature','doy_series']}
    
    def _apply_model(self, temperature, doy_series, t1, T, F):
        #Temperature threshold
        temperature[temperature<T]=0
    
        #Only accumulate forcing after t1
        temperature[doy_series<t1]=0
    
        accumulated_gdd=utils.forcing_accumulator(temperature)
    
        return utils.doy_estimator(forcing = accumulated_gdd, 
                                   doy_series = doy_series, 
                                   threshold = F)

class M1(_base_model):
    """The Thermal Time Model with a daylenght correction.
    
    Blümel & Chmielewski 2012
    
    Event happens on :math:`DOY` when the following is met:
    
    .. math::
        \sum_{t=t_{1}}^{DOY}R_{f}(T_{i})\geq (\frac{L_{i}}{24})^{k} F^{*}
    
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
        _base_model.__init__(self)
        self.all_required_parameters = {'t1':(-67,298),'T':(-25,25),
                                        'F':(0,1000),'k':(0,50)}
        self._organize_parameters(parameters)
        self._required_data={'predictor_columns':['site_id','year','doy','temperature', 'daylength'],
                             'predictors':['temperature','doy_series','daylength']}
    
    def _organize_predictors(self, predictors, observations, for_prediction):
        """Convert data to internal structure used by M1 model
        """
        # daylength for each observation. 1d array with length n, which should match
        # axis 0 of temperature_array
        obs_with_daylength = observations.merge(predictors, on=['site_id','year'], how='left')
        daylength_array = obs_with_daylength.daylength.values
        
        if for_prediction:
            temperature_array, doy_series = utils.temperature_only_data_prep(observations=observations, 
                                                                             predictors=predictors,
                                                                             for_prediction=for_prediction)
            return {'temperature': temperature_array,
                    'daylength' :  daylength_array,
                    'doy_series':  doy_series}
        else:
            cleaned_observations, temperature_array, doy_series = utils.temperature_only_data_prep(observations, 
                                                                                                   predictors,
                                                                                                   for_prediction=for_prediction)
            self.fitting_predictors = {'temperature' : temperature_array,
                                       'daylength' :   daylength_array,
                                       'doy_series' :  doy_series}
            
            self.obs_fitting = cleaned_observations
    
    def _validate_formatted_predictors(self, predictors):
        pass
    
    def _apply_model(self, temperature, doy_series, daylength, t1, T, F, k):
        #Temperature threshold
        temperature[temperature<T]=0
    
        #Only accumulate forcing after t1
        temperature[doy_series<t1]=0
    
        accumulated_gdd=utils.forcing_accumulator(temperature)
    
        daylength_adjustment = (daylength / 24) ** k
        # Make it the same shape as gdd for easy adjustment
        num_days = len(doy_series)
        daylength_adjustment = np.tile(np.expand_dims(daylength_adjustment, 1), num_days)
    
        return utils.doy_estimator(forcing = accumulated_gdd, 
                                   doy_series = doy_series, 
                                   threshold = F)