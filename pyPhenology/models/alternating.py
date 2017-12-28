import numpy as np
from . import utils
from .base import _base_model

class Alternating(_base_model):
    """Alternating model, originally defined in Cannell & Smith 1983.
    Phenological event happens the first day that forcing is greater 
    than an exponential curve of number of chill days.
    
    Parameters
    ----------
    a : int | float
        Intercept of chill day curve
    
    b : int | float
        Slope of chill day curve
    
    c : int | float
        scale parameter of chill day curve
        
    threshold : int | flaot
        Degree threshold above which forcing accumulates, and
        below which chilling accumulates. Set to 5 (assuming C)
        by default.
        
    t1 : int
        DOY which forcing and chilling accumulationg starts. Set
        to 1 (Jan 1) by default.
    """
    def __init__(self, parameters={}):
        _base_model.__init__(self)
        self.all_required_parameters = {'a':(-1000,1000), 'b':(0,5000), 'c':(-5,0),
                                        'threshold':(5,5), 't1':(1,1)}
        self._organize_parameters(parameters)
    
    def _apply_model(self, temperature, doy_series, a, b, c, threshold, t1):
        chill_days = ((temperature < threshold)*1).copy()
        chill_days[doy_series < t1]=0
        chill_days = utils.forcing_accumulator(chill_days)

        # Accumulated growing degree days from Jan 1
        gdd = temperature.copy()
        gdd[gdd < threshold]=0
        gdd[doy_series < t1]=0
        gdd = utils.forcing_accumulator(gdd)

        # Phenological event happens the first day gdd is > chill_day curve
        chill_day_curve = a + b * np.exp( c * chill_days)
        difference = gdd - chill_day_curve

        # The estimate is equal to the first day that
        # gdd - chill_day_curve > 0
        return utils.doy_estimator(difference, doy_series, threshold=0)
    
class MSB(_base_model):
    """Macroscale Species-specific Budburst model. Jeong et al. 2013
    Extension of the Alternating model which add a correction (d)
    using the mean spring temperature 
    
    Parameters
    ----------
    a : int | float
        Intercept of chill day curve
    
    b : int | float
        Slope of chill day curve
    
    c : int | float
        scale parameter of chill day curve
    
    d : int | float
        Correction factor
        
    threshold : int | flaot
        Degree threshold above which forcing accumulates, and
        below which chilling accumulates. Set to 5 (assuming C)
        by default.
        
    t1 : int
        DOY which forcing and chilling accumulationg starts. Set
        to 1 (Jan 1) by default.
    """
    def __init__(self, parameters={}):
        _base_model.__init__(self)
        self.all_required_parameters = {'a':(-1000,1000), 'b':(0,5000), 'c':(-5,0),
                                        'd':(-100,100), 'threshold':(5,5), 't1':(1,1)}
        self._organize_parameters(parameters)
    
    def _apply_model(self, temperature, doy_series, a, b, c, d, threshold, t1):
        chill_days = ((temperature < threshold)*1).copy()
        chill_days[doy_series < t1]=0
        chill_days = utils.forcing_accumulator(chill_days)

        # Accumulated growing degree days from Jan 1
        gdd = temperature.copy()
        gdd[gdd < threshold]=0
        gdd[doy_series < t1]=0
        gdd = utils.forcing_accumulator(gdd)

        chill_day_curve = a + b * np.exp( c * chill_days)
        
        # Make the spring temps the same shape as chill_day_curve
        # for easy addition.
        mean_spring_temp = utils.mean_temperature(temperature, doy_series,
                                                  start_doy=0, end_doy=60)
        mean_spring_temp *= d
        # Add in correction based on per site spring temperature
        chill_day_curve += mean_spring_temp

        # Phenological event happens the first day gdd is > chill_day curve
        difference = gdd - chill_day_curve

        # The estimate is equal to the first day that
        # gdd - chill_day_curve > 0
        return utils.doy_estimator(difference, doy_series, threshold=0)