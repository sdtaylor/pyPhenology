from . import utils
from .base import _base_model

class Uniforc(_base_model):
    """Single phase forcing model using a 
    sigmoid function for forcing units.
    Chuine 2000
    
    Parameters
    ----------
    t1 : int
        The DOY which forcing accumulating beings
    
    F : int, > 0
        The total forcing units required
        
    b : int
        Sigmoid function parameter
    
    c : int
        Sigmoid function parameter
    """
    def __init__(self, parameters={} ):
        _base_model.__init__(self)
        self.all_required_parameters = {'t1':(-67,298),'F':(0,200),'b':(-20,0),'c':(-50,50)}
        self._organize_parameters(parameters)
    
    def _apply_model(self, temperature, doy_series, t1, F, b, c):
        temperature = utils.sigmoid2(temperature, b=b, c=c)
    
        #Only accumulate forcing after t1
        temperature[doy_series<t1]=0
    
        accumulateed_forcing=utils.forcing_accumulator(temperature)
    
        return utils.doy_estimator(forcing = accumulateed_forcing,
                                   doy_series=doy_series,
                                   threshold=F)

class Unichill(_base_model):
    """Two phase forcing model using a 
    sigmoid function for forcing units 
    and chilling. 
    Chuine 2000
    
    Parameters
    ----------
    t0 : int
        The DOY which chilling accumulating beings
    
    C : int, > 0
        The total chilling units required

    F : int, > 0
        The total forcing units required
        
    b_f : int
        Sigmoid function parameter for forcing
    
    c_f : int
        Sigmoid function parameter for forcing
        
    a_c : int
        Sigmoid funcion parameter for chilling
        
    b_c : int
        Sigmoid funcion parameter for chilling
        
    c_c : int
        Sigmoid funcion parameter for chilling
    """
    def __init__(self, parameters={}):
        _base_model.__init__(self)
        self.all_required_parameters = {'t0':(-67,298),'C':(0,300),'F':(0,200),
                                        'b_f':(-20,0),'c_f':(-50,50),
                                        'a_c':(0,20),'b_c':(-20,20),'c_c':(-50,50)}
        self._organize_parameters(parameters)
    
    def _apply_model(self, temperature, doy_series, t0, C, F, b_f, c_f, a_c, b_c, c_c):
        assert len(temperature.shape)==2, 'Unichill model currently only supports 2d temperature arrays'

        temp_chilling = temperature.copy()
        temp_forcing  = temperature.copy()
        
        temp_forcing = utils.sigmoid2(temp_forcing, b=b_f, c=c_f)
        temp_chilling =utils.sigmoid3(temp_chilling, a=a_c, b=b_c, c=c_c) 
    
        #Only accumulate chilling after t0
        temp_chilling[doy_series<t0]=0
        accumulated_chill=utils.forcing_accumulator(temp_chilling)
        
        # Heat forcing accumulation starts when the chilling
        # requirement, C, has been met(t1 in the equation). 
        # Enforce this by setting everything prior to that date to 0
        # TODO: optimize this so it doesn't use a for loop
        t1_values = utils.doy_estimator(forcing = accumulated_chill,
                                        doy_series=doy_series,
                                        threshold=C)
        for col, t1 in enumerate(t1_values):
            temp_forcing[doy_series<t1, col]=0
    
        accumulated_forcing = utils.forcing_accumulator(temp_forcing)
        
        return utils.doy_estimator(forcing = accumulated_forcing,
                                   doy_series=doy_series,
                                   threshold=F)