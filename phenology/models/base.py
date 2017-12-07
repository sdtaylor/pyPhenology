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
        self.temperature_fitting, self.doy_series = utils.format_temperature(DOY, temperature)
        
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
        
    def predict(self, site_years, temperature, return_type='array'):
        # utils.validate_temperature
        assert len(self._fitted_params) > 0, 'Parameters not set'
        
        utils.validate_temperature(temperature)
        utils.validate_DOY(site_years, for_prediction=True)
        temp_array, doy_series = utils.format_temperature(site_years, temperature)
        
        predictions = self._apply_model(temp_array, doy_series, **self._fitted_params)
        
        if return_type == 'array':
            return predictions
        elif return_type == 'df':
            site_years['doy_predicted'] = predictions
            return site_years
        
    def get_params(self):
        return self._fitted_params
    
    def set_params(self, params):
        for p in params:
            assert p in self.parameters, 'unknown parameter: ' + str(p)
        self._fitted_params = params

    def get_initial_bounds(self):
        return self.bounds
    
    def get_doy_estimates(self, **params):
        return self._apply_model(temperature = self.temperature_fitting, 
                                 doy_series = self.doy_series,
                                 **params)
    
    def get_error(self, **kargs):
        doy_estimates = self.get_doy_estimates(**kargs)
        error = np.sqrt(np.mean((doy_estimates - self.DOY_fitting)**2))
        return error
    
    def _translate_scipy_parameters(self, parameters_array):
        """Map paramters from a 1D array to a dictionary for
        use in phenology models
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
    def __init__(self):
        _base_model.__init__(self)
        self.bounds = {'t1':(-67,298), 
                       'T':(-25,25),
                       'F':(0,1000)}
        self.parameters = ['t1','T','F']
    
    def _apply_model(self, temperature, doy_series, t1, T, F):
        #Temperature cutoff
        temperature[temperature<T]=0
    
        #Only accumulate forcing after t1
        temperature[:,doy_series<t1]=0
    
        accumulated_gdd=utils.forcing_accumulator(temperature)
    
        return utils.doy_estimator(forcing = accumulated_gdd, 
                                   doy_series = doy_series, 
                                   threshold = F)


