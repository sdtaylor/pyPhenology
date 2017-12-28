from . import utils
from .base import _base_model

class ThermalTime(_base_model):
    """The classic growing degree day model using
    a fixed threshold above which forcing accumulates.
    
    Parameters
    ----------
    t1 : int
        The DOY which forcing accumulating beings
    
    T : int
        The threshold above which forcing accumulates
    
    F : int, > 0
        The total forcing units required
    """
    def __init__(self, parameters={}):
        _base_model.__init__(self)
        self.all_required_parameters = {'t1':(-67,298),'T':(-25,25),'F':(0,1000)}
        self._organize_parameters(parameters)
    
    def _apply_model(self, temperature, doy_series, t1, T, F):
        #Temperature threshold
        temperature[temperature<T]=0
    
        #Only accumulate forcing after t1
        temperature[doy_series<t1]=0
    
        accumulated_gdd=utils.forcing_accumulator(temperature)
    
        return utils.doy_estimator(forcing = accumulated_gdd, 
                                   doy_series = doy_series, 
                                   threshold = F)