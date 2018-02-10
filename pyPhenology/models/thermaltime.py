from . import utils
from .base import _base_model

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
    
    def _apply_model(self, temperature, doy_series, t1, T, F):
        #Temperature threshold
        temperature[temperature<T]=0
    
        #Only accumulate forcing after t1
        temperature[doy_series<t1]=0
    
        accumulated_gdd=utils.forcing_accumulator(temperature)
    
        return utils.doy_estimator(forcing = accumulated_gdd, 
                                   doy_series = doy_series, 
                                   threshold = F)
