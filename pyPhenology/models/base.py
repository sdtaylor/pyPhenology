import numpy as np
from . import utils
from scipy import optimize

class _base_model():
    def __init__(self):
        self._fitted_params = {}
        
    def fit(self, DOY, temperature, method='DE', verbose=False):
        utils.validate_temperature(temperature)
        utils.validate_DOY(DOY)
        self.DOY_fitting = DOY.doy.values
        self.temperature_fitting, self.doy_series = utils.format_temperature(DOY, temperature, verbose=verbose)
        
        # TODO: make this it's own function or class to allow other methods
        # like basinhopping or brute force and configurable params
        optimize_output = optimize.differential_evolution(self._scipy_error,
                                                          bounds=self._scipy_bounds(), 
                                                          disp=verbose, 
                                                          maxiter=20, 
                                                          popsize=10, 
                                                          mutation=1.5, 
                                                          recombination=0.25)
        
        self._fitted_params = self._translate_scipy_parameters(optimize_output['x'])
        
    def predict(self, site_years=None, temperature=None, return_type='array'):
        # utils.validate_temperature
        assert len(self._fitted_params) > 0, 'Parameters not set'
        
        utils.validate_temperature(temperature)
        utils.validate_DOY(site_years, for_prediction=True)
        temp_array, doy_series = utils.format_temperature(site_years, temperature)
        
        predictions = self._apply_model(temp_array.copy(),
                                        doy_series.copy(),
                                        **self._fitted_params)
        
        if return_type == 'array':
            return predictions
        elif return_type == 'df':
            site_years['doy_predicted'] = predictions
            return site_years
    
    def set_predict(self):
        pass
        # set parameters and predict in one go
        
    def get_params(self):
        return self._fitted_params
    
    def set_params(self, params):
        for p in params:
            assert p in self.parameters, 'unknown parameter: ' + str(p)
        self._fitted_params = params

    def get_initial_bounds(self):
        return self.bounds
    
    def get_doy_estimates(self, **params):
        return self._apply_model(temperature = self.temperature_fitting.copy(), 
                                 doy_series = self.doy_series.copy(),
                                 **params)
    
    def get_error(self, **kargs):
        doy_estimates = self.get_doy_estimates(**kargs)
        error = np.sqrt(np.mean((doy_estimates - self.DOY_fitting)**2))
        return error
    
    def _translate_scipy_parameters(self, parameters_array):
        """Map paramters from a 1D array to a dictionary for
        use in phenology model functions.
        """
        labeled_parameters={}
        for i, param in enumerate(self.parameters):
            labeled_parameters[param]=parameters_array[i]
        return labeled_parameters
    
    def _scipy_error(self,x):
        """Error function for use within scipy.optimize functions.        
        """
        parameters = self._translate_scipy_parameters(x)
        return self.get_error(**parameters)
    
    def _scipy_bounds(self):
        """Bounds structured for scipy.optimize input"""
        return [self.bounds[param] for param  in self.parameters]
    
    def score(self, metric='rmse'):
        pass

    
class Thermal_Time(_base_model):
    """The classic growing degree day model using
    a fixed threshold above which forcing accumulates.
    
    Parameters
    ----------
    t1 : int
        The doy which forcing accumulating beings
    
    T : int
        The threshold above which forcing accumulates
    
    F : int, > 0
        The total forcing units required
    """
    def __init__(self):
        _base_model.__init__(self)
        self.bounds = {'t1':(-67,298), 
                       'T':(-25,25),
                       'F':(0,1000)}
        self.parameters = ['t1','T','F']
    
    def _apply_model(self, temperature, doy_series, t1, T, F):
        #Temperature threshold
        temperature[temperature<T]=0
    
        #Only accumulate forcing after t1
        temperature[:,doy_series<t1]=0
    
        accumulated_gdd=utils.forcing_accumulator(temperature)
    
        return utils.doy_estimator(forcing = accumulated_gdd, 
                                   doy_series = doy_series, 
                                   threshold = F)

class Uniforc(_base_model):
    """Single phase forcing model using a 
    sigmoid function for forcing units.
    Chuine 2000
    
    Parameters
    ----------
    t1 : int
        The doy which forcing accumulating beings
    
    F : int, > 0
        The total forcing units required
        
    b : int
        Sigmoid function parameter
    
    c : int
        Sigmoid function parameter
    """
    def __init__(self):
        _base_model.__init__(self)
        self.bounds = {'t1':(-67,298), 
                       'F':(0,200),
                       'b':(-20,0),
                       'c':(-100,100)}
        self.parameters = ['t1','F','b','c']
    
    def _apply_model(self, temperature, doy_series, t1, F, b, c):
        temperature = utils.sigmoid2(temperature, b=b, c=c)
    
        #Only accumulate forcing after t1
        temperature[:,doy_series<t1]=0
    
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
        The doy which chilling accumulating beings
    
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
    def __init__(self):
        _base_model.__init__(self)
        self.bounds = {'t0':(-67,298), 
                       'C':(0,300),
                       'F':(0,200),
                       'b_f':(-20,0),
                       'c_f':(-100,100),
                       'a_c':(0,20),
                       'b_c':(-100,100),
                       'c_c':(-50,50)}
        self.parameters = ['t0','C','F','b_f','c_f','a_c','b_c','c_c']
    
    def _apply_model(self, temperature, doy_series, t0, C, F, b_f, c_f, a_c, b_c, c_c):
        temp_chilling = temperature.copy()
        temp_forcing  = temperature.copy()
        
        temp_forcing = utils.sigmoid2(temp_forcing, b=b_f, c=c_f)
        temp_chilling =utils.sigmoid3(temp_chilling, a=a_c, b=b_c, c=c_c) 
    
        #Only accumulate chilling after t0
        temp_chilling[:,doy_series<t0]=0
        accumulated_chill=utils.forcing_accumulator(temp_chilling)
        
        #Heat forcing accumulation starts when the chilling
        # requirement, C, has been met. Enforce this by 
        # setting everything prior to that date to 0
        F_begin = utils.doy_estimator(forcing = accumulated_chill,
                                      doy_series=doy_series,
                                      threshold=C)
        for row in range(F_begin.shape[0]):
            temp_forcing[row, doy_series<F_begin[row]]=0
    
        accumulated_forcing = utils.forcing_accumulator(temp_forcing)
        
        return utils.doy_estimator(forcing = accumulated_forcing,
                                   doy_series=doy_series,
                                   threshold=F)
