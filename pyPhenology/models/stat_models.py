from . import utils
from .base import _base_model

class Linear(_base_model):
    """A linear regression where DOY ~ mean_spring_tempearture
    
    Parameters
    ----------
    intercept : int | float
        y intercept of the model
    
    slope : int | float
        Slope of the model
        
    spring_start : int
        The start day of spring, defaults to Jan 1 (DOY 0)
    
    spring_end : int
        The last day of spring, defaults to March 30 (DOY 90)

    """
    def __init__(self, parameters={}):
        _base_model.__init__(self)
        self.all_required_parameters = {'intercept':(-67,298),'slope':(-25,25),
                                        'spring_start':(0,0), 'spring_end':(90,90)}
        self._organize_parameters(parameters)
    
    def _apply_model(self, temperature, doy_series, intercept, slope, 
                     spring_start, spring_end):
        mean_spring_temp = utils.mean_temperature(temperature, doy_series,
                                                  start_doy = spring_start,
                                                  end_doy = spring_end)
        return mean_spring_temp * slope + intercept