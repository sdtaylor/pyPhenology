import numpy as np
import pandas as pd
from . import utils, validation
from copy import deepcopy


class BootstrapModel():
    """Fit a model using bootstrapping of the data.
    
    Bootstrapping is a technique to estimate model uncertainty. Many
    models of the same form are fit, but each use a random selection,
    with replacement, of the data.
    
    Note that the core model must be passed uninitialzed::
        
        from pyPhenology import models
        
        thermaltime_bootstrapped = models.BootstrapModel(core_model = models.ThermalTime)
        
    Starting parameters for the core model can be adjusted as normal.
    For example to fix the start day of the ThermalTime model::
    
        thermaltime_bootstrapped = models.BootstrapModel(core_model = models.ThermalTime,
                                                         parameters = {'t1':1})
    

    """
    def __init__(self, core_model, num_bootstraps, parameters={}):
        """Bootstrap Model
        
        Parameters:
            core_model : pyPhenology model
                The model to fit n number of times. Must be uninitialized
            
            num_bootstraps : int
                Number of times to fit the model
            
            parameters : dictionary | filename, optional
                Parameter search ranges or fixed values for the core model.
                If a filename, then it must be a bootstrap model save via
                ``save_params()``
        
        """
        
        
        validation.validate_model(core_model())
        
        self.model_list=[]
        if isinstance(parameters, str):
            # A filename pointing toward a file from save_params()
            params = pd.read_csv(parameters).to_dict('records')
            for bootstrap_iteration in params:
                bootstrap_iteration.pop('bootstrap_num')
                self.model_list.append(core_model(parameters=bootstrap_iteration))
        elif isinstance(parameters, dict):
            for i in range(num_bootstraps):            
                self.model_list.append(core_model(parameters=parameters))
        elif isinstance(parameters, list):
            # If its the output of BootstrapModel.get_params()
            for bootstrap_iteration in parameters:
                bootstrap_iteration.pop('bootstrap_num')
                self.model_list.append(core_model(parameters=bootstrap_iteration))
        else:
            raise TypeError('parameters must be str or dict, got: ' + str(type(parameters)))
    
    def fit(self,observations, temperature, **kwargs):
        """Fit the underlying core models"""
        #TODO: do the temperature transform here cause so it doesn't get reapated a bunch
        # need to wait till fit takes arrays directly
        validation.validate_observations(observations)
        validation.validate_temperature(temperature)
        for model in self.model_list:
            obs_shuffled = observations.sample(frac=1, replace=True).copy()
            model.fit(obs_shuffled, temperature, **kwargs)

    def predict(self,to_predict=None, temperature=None, aggregation='mean', **kwargs):
        """Make predictions from the bootstrapped models.
        
        Predictions will be made using each of the bootstrapped models.
        The final results will be the mean or median of all bootstraps, or all bootstrapped
        model results in 2d array.
        
        Parameters:
            aggregation : str
                Either 'mean','median', or 'none'. 'none' return *all* predictions
                in an array of size (num_bootstraps, num_samples)
        
        """
        #TODO: do the temperature transform here cause so it doesn't get reapated a bunch
        # need to wait till predict takes arrays directly
        predictions=[]
        for model in self.model_list:
            predictions.append(model.predict(to_predict=to_predict, 
                                             temperature=temperature,
                                             **kwargs))
        
        predictions = np.array(predictions)
        if aggregation=='mean':
            predictions = np.mean(predictions, 0)
        elif aggregation=='median':
            predictions = np.median(predictions, 0)
        elif aggregation=='none':
            pass
        else:
            raise ValueError('Unknown aggregation: ' + str(aggregation))
        
        return predictions

    def get_params(self):
        
        all_params=[]
        for i, model in enumerate(self.model_list):
            all_params.append(deepcopy(model.get_params()))
            all_params[-1].update({'bootstrap_num':i})

        return all_params

    def save_params(self, filename):
        """Save model parameters
        
        Note this will save details on all bootstrapped models, and 
        can only be loaded again as a bootstrap model.
        
        Parameters:
            filename : str
                filename to save model to.
        
        """
        
        if len(self.model_list[0]._fitted_params)==0:
            raise RuntimeError('Parameters not fit, nothing to save')
        params = self.get_params()
        pd.DataFrame(params).to_csv(filename, index=False)
