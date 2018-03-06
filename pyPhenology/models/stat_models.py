from . import utils
from .base import _base_model

class Linear(_base_model):
    """Linear Regression Model
    
    A 2 parameter regression model with :math:`DOY` as
    the response variable. 

    .. math::
        DOY = \\beta_{1} + \\beta_{2}T_{mean}
    
    where :math:`T_{mean}` is the mean spring temperature.
    The start and end of spring is configurable.
    
    Parameters:
        intercept : int | float
            | :math:`\\beta_{1}`, intercept of the model
            | default : (-67,298)
        
        slope : int | float
            | :math:`\\beta_{1}`, Slope of the model
            | default : (-25,25)
            
        spring_start : int
            | The start day of spring
            | default : 1 (Jan 1)
        
        spring_end : int
            | The last day of spring
            | default : 90 \(March 30\)

    """
    def __init__(self, parameters={}):
        _base_model.__init__(self)
        self.all_required_parameters = {'intercept':(-67,298),'slope':(-25,25),
                                        'spring_start':0, 'spring_end':90}
        self._organize_parameters(parameters)
        self._required_data={'predictor_columns':['site_id','year','doy','temperature'],
                             'predictors':['temperature','doy_series']}
        
    def _apply_model(self, temperature, doy_series, intercept, slope, 
                     spring_start, spring_end):
        mean_spring_temp = utils.mean_temperature(temperature, doy_series,
                                                  start_doy = spring_start,
                                                  end_doy = spring_end)
        return mean_spring_temp * slope + intercept

