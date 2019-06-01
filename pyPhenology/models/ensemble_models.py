import numpy as np
from . import utils
from ..utils import load_model, load_model_parameters
from . import validation
from copy import deepcopy
import warnings
from joblib import Parallel, delayed

class EnsembleBase():
    def __init__(self):
        pass
    
    def score(self, metric='rmse', doy_observed=None,
              to_predict=None, predictors=None):
        """Get the scoring metric for fitted data
        Get the score on the dataset used for fitting (if fitting was done),
        otherwise set ``to_predict``, and ``predictors`` as used in
        ``model.predict()``. In the latter case score is calculated using
        observed values ``doy_observed``.
        Metrics available are root mean square error (``rmse``).
        Parameters:
            metric : str
                Currently only rmse is available for BootstrapModel
        """
        self._check_parameter_completeness()
        doy_estimated = self.predict(to_predict=to_predict,
                                     predictors=predictors)

        if doy_observed is None:
            doy_observed = self.observations.doy.values
        elif isinstance(doy_observed, np.ndarray):
            pass
        else:
            raise TypeError('Unknown doy_observed parameter type. expected ndarray, got ' + str(type(doy_observed)))

        error_function = utils.optimize.get_loss_function(method=metric)

        return error_function(doy_observed, doy_estimated)
    
    def save_params(self, filename, overwrite=False):
        """Save model parameters
        

        Parameters:
            filename : str
                Filename to save model to. Note this can be loaded again by
                passing the filename in the ``parameters`` argument, but only
                with the BootstrapModel.
            overwrite : bool
                Overwrite the file if it exists
        """
        self._check_parameter_completeness()

        utils.misc.write_saved_model(model_info=self._get_model_info(),
                                     model_file=filename,
                                     overwrite=overwrite)

    def _fit_job(self, model, **kwargs):
        """A wrapper to fit models within joblib.Parallel"""
        model.fit(self.observations, self.predictors, **kwargs)
        return model
    
    def _predict_job(self, model, to_predict, predictors, **kwargs):
        """A wrapper to predict new data within joblib.Parallel"""
        return model.predict(to_predict=to_predict, predictors=predictors, **kwargs)
    
    def ensemble_shape(self, shape=()):
        """Returns a tuple signifying the layers of submodels
        ie. for a single 50-bootstrap model the shape is (50,)
        for an ensemble of four 50-bootstrapped  models the shape is (4,50)
        """
        num_sub_models = len(self.model_list)
        if isinstance(self.model_list[0], EnsembleBase):
            return self.model_list[0].ensemble_shape(shape = shape + (num_sub_models,))
        else:
            return shape + (num_sub_models,)
    
class BootstrapModel(EnsembleBase):
    """Fit a model using bootstrapping of the data.

    Bootstrapping is a technique to estimate model uncertainty. Many
    models of the same form are fit, but each use a random selection,
    with replacement, of the data.

    Note that the core model must be passed uninitialized::

        from pyPhenology import models

        thermaltime_bootstrapped = models.BootstrapModel(core_model = models.ThermalTime)

    Starting parameters for the core model can be adjusted as normal.
    For example to fix the start day of the ThermalTime model::

        thermaltime_bootstrapped = models.BootstrapModel(core_model = models.ThermalTime,
                                                         parameters = {'t1':1})


    """

    def __init__(self, core_model=None, num_bootstraps=None, parameters={}):
        """Bootstrap Model

        Parameters:
            core_model : pyPhenology model
                The model to fit n number of times. Must be uninitialized

            num_bootstraps : int
                Number of times to fit the model

            parameters : dictionary | filename, optional
                Parameter search ranges or fixed values for the core model.
                If a filename, then it must be a bootstrap model saved via
                ``save_params()``.
                Also if it is a saved model file then the core_model and
                num_bootstrap parameters are ignored.
        """
        EnsembleBase.__init__(self)
        self.observations = None
        self.predictors = None
        self.model_list = []
        if isinstance(parameters, str):
            # A filename pointing toward a file from save_params()
            # core_model and num_bootstraps is ignored
            model_info = utils.misc.read_saved_model(model_file=parameters)
            self._parse_fully_fitted_model(model_info)
 
        elif isinstance(parameters, dict):
            # A dictionary passed in the parameters argument can either be
            # a list of parameters to pass to the core model (such as in the 
            # example above), OR a full fitted model specification (essentially
            # the read json from a saved model file). The latter is only used 
            # if this is from another ensemble method
            
            if 'model_name' in parameters:
                # A saved model
                self._parse_fully_fitted_model(parameters)
            else:
                # Custom parameter values to pass to each bootstrap model
                if core_model is None or num_bootstraps is None:
                    raise TypeError('core_model and num_bootstraps must be set')
    
                validation.validate_model(core_model())
                for i in range(num_bootstraps):
                    self.model_list.append(core_model(parameters=parameters))

        elif isinstance(parameters, list):
            # If its the output of BootstrapModel._get_model_info()
            for bootstrap_iteration in parameters:
                bootstrap_iteration.pop('bootstrap_num')
                self.model_list.append(core_model(parameters=bootstrap_iteration))
        else:
            raise TypeError('parameters must be str or dict, got: ' + str(type(parameters)))

    def _parse_fully_fitted_model(self, model_info):
        # This loads a model from a saved json file
        if type(self).__name__ != model_info['model_name']:
            raise RuntimeError('Saved model file does not match model class. ' +
                               'Saved file is {a}, this model is {b}'.format(a=model_info['model_name'],
                                                                             b=type(self).__name__))

        for bootstrap_iteration in model_info['parameters']:
            fitted_bootstrap_iteration = load_model_parameters(bootstrap_iteration)
            self.model_list.append(fitted_bootstrap_iteration)
    
    def _fit_job(self, model, **kwargs):
        """A wrapper to fit models within joblib.Parallel"""
        obs_shuffled = self.observations.sample(frac=1, replace=True).copy()
        model.fit(obs_shuffled, self.predictors, **kwargs)
        return model

    
    def fit(self, observations, predictors, n_jobs=1, verbose=False, debug=False, **kwargs):
        """Fit the underlying core models

        Parameters:
            observations : dataframe
                pandas dataframe of phenology observations

            predictors : dataframe
                pandas dataframe of associated predictors

            n_jobs : int
                number of parallel processes to use

            kwargs :
                Other arguments passed to core model fitting (eg. optimzer methods)
        """
        self.observations = observations
        self.predictors = predictors
        
        self.model_list = Parallel(n_jobs = n_jobs)(delayed(self._fit_job)(m, **kwargs) for m in self.model_list)

    def predict(self, to_predict=None, predictors=None, 
                aggregation='mean', n_jobs=1, **kwargs):
        """Make predictions from the bootstrapped models.

        Predictions will be made using each of the bootstrapped models.
        The final results will be the mean or median of all bootstraps, or all bootstrapped
        model results in 2d array.

        Parameters:
            aggregation : str
                Either 'mean','median', or 'none'. 'none' return *all* predictions
                in an array of size (num_bootstraps, num_samples)
            
            n_jobs : int
                number of parallel processes to use

        """
        # Get the organized predictors. This is done from the 1st model in the
        # list, but since they're all the same model it should be valid for all.
        # Only works if there are new predictors. Otherwise the data used for
        # fitting will be  used.
        # TODO: implement this
        # if predictors is not None:
        #   predictors = self.model_list[0]._organize_predictors(observations=to_predict,
        #                                                         predictors=predictors,
        #                                                         for_prediction=True)
        if predictors is None:
            predictors = self.predictors
        if to_predict is None:
            to_predict = self.observations

        # If the models within this ensemble are themselves ensembles,
        # then let those models do the parallel stuff.
        if isinstance(self.model_list[0], EnsembleBase):
            predictions = []
            for m in self.model_list:
                predictions.append(m.predict(to_predict = to_predict,
                                             predictors = predictors,
                                             aggregation = aggregation,
                                             n_jobs = n_jobs,
                                             **kwargs))
        else:
            predictions = Parallel(n_jobs = n_jobs)(delayed(self._predict_job)
                (m, to_predict = to_predict, predictors = predictors, 
                 aggregation=aggregation, **kwargs)
                for m in self.model_list)

        predictions = np.array(predictions)
        if aggregation == 'mean':
            predictions = np.mean(predictions, 0)
        elif aggregation == 'median':
            predictions = np.median(predictions, 0)
        elif aggregation == 'none':
            pass
        else:
            raise ValueError('Unknown aggregation: ' + str(aggregation))

        return predictions

    def _check_parameter_completeness(self):
        """Make sure all parameters have been set from fitting or loading at initialization"""
        
        [m._check_parameter_completeness() for m in self.model_list]

    def get_params(self):
        """This returns list of dictionaries with parameters of each bootstrap model
        """
        self._check_parameter_completeness()
        
        all_params = []
        for i, model in enumerate(self.model_list):
            all_params.append(deepcopy(model.get_params()))
            all_params[-1].update({'bootstrap_num': i})

        return all_params

    def _get_model_info(self):
        # essentially the same as get_params() but is in a formate capable of
        # being loaded again later by _parse_fully_fitted_model()
        core_model_info = []
        for i, model in enumerate(self.model_list):
            core_model_info.append(deepcopy(model._get_model_info()))
            core_model_info[-1].update({'bootstrap_num': i})
            
        return {'model_name': type(self).__name__,
                'parameters': core_model_info}

class Ensemble(EnsembleBase):
    """Fit an ensemble of different models.
    
    This model can fit multiple models into an ensemble where the 
    weights are equal among all ensemble members.
    
    Note that the core models must be passed initialized. They will be
    fit within the Ensemble model::

                from pyPhenology import models, utils
                observations, predictors = utils.load_test_data(name='vaccinium')
                
                m1 = models.Thermaltime(parameters={'T':0})
                m2 = models.Thermaltime(parameters={'T':5})
                m3 = models.Uniforc(parameters={'t1':1})
                m4 = models.Uniforc(parameters={'t1':30})

                ensemble = models.Ensemble(core_models=[m1,m2,m3,m4])
                ensemble.fit(observations, predictors)
    
    """
    def __init__(self, core_models):
        """Ensemble model
        
        core_models : list of pyPhenology models, or a saved model file

        """
        EnsembleBase.__init__(self)
        self.observations = None
        self.predictors = None

        if isinstance(core_models, list):
            # List of models to fit
            self.model_list = core_models
            
        elif isinstance(core_models, str):
            # A filename pointing toward a file from save_params()
            model_info = utils.misc.read_saved_model(model_file=core_models)
            self._parse_fully_fitted_model(model_info)

        elif isinstance(core_models, dict):
            # A saved model file read by another model or utility
            self._parse_fully_fitted_model(core_models)

        else:
            raise TypeError('core_models must be list of pyPhenology models',
                            'or a filename for a saved model')

    def _parse_fully_fitted_model(self, model_info):
        # This loads a model from a saved json file
        if type(self).__name__ != model_info['model_name']:
            raise RuntimeError('Saved model file does not match model class. ',
                               'Saved file is {a}, this model is {b}'.format(a=model_info['model_name'],
                                                                             b=type(self).__name__))

        self.model_list = []
        for model in model_info['core_models']:
            fitted_ensemble_member = load_model_parameters(model)
            self.model_list.append(fitted_ensemble_member)

    def fit(self, observations, predictors, n_jobs=1, verbose=False, debug=False, **kwargs):
        """Fit the underlying core models
        Parameters:
            observations : dataframe
                pandas dataframe of phenology observations

            predictors : dataframe
                pandas dataframe of associated predictors

            n_jobs : int
                number of parallel processes to use

            kwargs :
                Other arguments passed to core model fitting (eg. optimzer methods)
        """
        self.observations = observations
        self.predictors = predictors

        self.model_list = Parallel(n_jobs = n_jobs)(delayed(self._fit_job)(m, **kwargs) for m in self.model_list)

    def predict(self, to_predict=None, predictors=None, 
                aggregation='mean', n_jobs=1, **kwargs):
        """Make predictions..

        Predictions will be made using each core models, then a final prediction
        for each observation using the specified aggregation method.

        Parameters:
            see core model description
            
            aggregation : str
                Either 'mean', 'median', or 'none'. If using 'none' this returns
                an array of tuple of size (number of members, predictions). 
            
            n_jobs : int
                number of parallel processes to use
        """
        if predictors is None:
            predictors = self.predictors
        if to_predict is None:
            to_predict = self.observations

        # If the models within this ensemble are themselves ensembles,
        # then let those models do the parallel stuff.
        if isinstance(self.model_list[0], EnsembleBase):
            predictions = []
            for m in self.model_list:
                predictions.append(m.predict(to_predict = to_predict,
                                             predictors = predictors,
                                             aggregation = aggregation,
                                             n_jobs = n_jobs,
                                             **kwargs))
        else:
            predictions = Parallel(n_jobs = n_jobs)(delayed(self._predict_job)
                (m, to_predict = to_predict, predictors = predictors, 
                 aggregation=aggregation, **kwargs)
                for m in self.model_list)

        predictions = np.array(predictions)
        if aggregation == 'mean':
            predictions = np.mean(predictions, 0)
        elif aggregation == 'median':
            predictions = np.median(predictions, 0)
        elif aggregation == 'none':
            pass
        else:
            raise ValueError('Unknown aggregation: ' + str(aggregation))

        return predictions

    def _check_parameter_completeness(self):
        """Make sure all parameters have been set from fitting or loading at initialization"""
        [m._check_parameter_completeness() for m in self.model_list]

    def get_params(self):
        """This returns list of dictionaries with parameters of each model in the ensemble
        """
        self._check_parameter_completeness()
        
        all_params = []
        for i, model in enumerate(self.model_list):
            all_params.append(deepcopy(model.get_params()))

        return all_params

    def _get_model_info(self):
        # essentially the same as get_params() but is in a formate capable of
        # being loaded again later by _parse_fully_fitted_model()
        core_model_info = []
        for i, model in enumerate(self.model_list):
            core_model_info.append(deepcopy(model._get_model_info()))

        return {'model_name': type(self).__name__,
                'core_models': core_model_info}

class WeightedEnsemble(EnsembleBase):
    """Fit an ensemble of many models with associated weights

    This model can combine multiple models into an ensemble where predictions
    are the weighted average of the predictions from each model. The weights
    are derived via "stacking" as described in Dormann et al. 2018. The 
    steps are as followed:

        1. Subset the data into random training/testing sets.
        2. Fit each core model on the training set.
        3. Make predictions on the testing set.
        4. Find the weights which minimize RMSE of the testing set. 
        5. Repeat 1-4 for H iterations.
        6. Take the average weight for each model from all iterations as
           final weight used in the ensemble. These will sum to 1. 
        7. Fit the core models a final time on the full dataset given
           to the fit() method. Parameters derived from this final
           iterations will be used to make predictions. 

    Note that the core models must be passed initialized. They will be
    fit within the Weighted Ensemble model::

                from pyPhenology import models, utils
                observations, predictors = utils.load_test_data(name='vaccinium')
                
                m1 = models.Thermaltime(parameters={'T':0})
                m2 = models.Thermaltime(parameters={'T':5})
                m3 = models.Thermaltime(parameters={'T':-5})
                m4 = models.Thermaltime(parameters={'T':10})
                m5 = models.Uniforc(parameters={'t1':1})
                m6 = models.Uniforc(parameters={'t1':30})
                m7 = models.Uniforc(parameters={'t1':60})
                
                ensemble = models.WeightedEnsemble(core_models=[m1,m2,m3,m4,m5,m6,m7])
                ensemble.fit(observations, predictors)

    Notes:
        Dormann, Carsten F., et al. Model averaging in ecology: a review of
        Bayesian, information‐theoretic and tactical approaches for predictive
        inference. Ecological Monographs. https://doi.org/10.1002/ecm.1309

    """
    def __init__(self, core_models):
        """Weighted Ensemble model
        
        core_models : list of pyPhenology models, or a saved model file

        """
        EnsembleBase.__init__(self)
        self.observations = None
        self.predictors = None

        if isinstance(core_models, list):
            # List of models to fit
            self.model_list = core_models
            self.weights = np.array([None] * len(core_models))
            
        elif isinstance(core_models, str):
            # A filename pointing toward a file from save_params()
            model_info = utils.misc.read_saved_model(model_file=core_models)
            self._parse_fully_fitted_model(model_info)

        elif isinstance(core_models, dict):
            # A saved model file read by another model or utility
            self._parse_fully_fitted_model(core_models)
        else:
            raise TypeError('core_models must be list of pyPhenology models',
                            'or a filename for a saved model')

    def _parse_fully_fitted_model(self, model_info):
        # This loads a model from a saved json file
        if type(self).__name__ != model_info['model_name']:
            raise RuntimeError('Saved model file does not match model class. ',
                               'Saved file is {a}, this model is {b}'.format(a=model_info['model_name'],
                                                                             b=type(self).__name__))

        self.model_list = []
        self.weights = []
        for model in model_info['core_models']:
            Model = load_model(model['model_name'])
            self.model_list.append(Model(parameters=model['parameters']))
            self.weights.append(model['weight'])

        self.weights = np.array(self.weights)

    def _fit_job(self, model, observations, predictors, **kwargs):
        """A wrapper to fit models within joblib.Parallel"""
        model.fit(observations, predictors, **kwargs)
        return model

    def fit(self, observations, predictors, iterations=10, held_out_percent=0.2,
            loss_function='rmse', method='DE', optimizer_params='practical',
            n_jobs=1, verbose=False, debug=False):
        """Fit the underlying core models
        
        Parameters:
            observations : dataframe
                pandas dataframe of phenology observations
            
            predictors : dataframe
                pandas dataframe of associated predictors
            
            iterations : int
                Number of stacking iterations to use.
            
            held_out_percent : float
                Percent of randomly held out data to use in each stacking
                iteration. Must be between 0 and 1. 

            n_jobs : int
                number of parallel processes to use

            kwargs :
                Other arguments passed to core model fitting (eg. optimzer methods)
        """
        self.observations = observations
        self.predictors = predictors
        
        self.fitted_weights = np.empty((iterations, len(self.model_list)))
        
        loss = utils.optimize.get_loss_function(loss_function)
        weight_bounds = [(1,10)] * len(self.model_list)
        translate_scipy_weights = lambda w: np.array(w)
        
        with Parallel(n_jobs = n_jobs) as parallel:
            for i in range(iterations):
                held_out_observations = self.observations.sample(frac=held_out_percent,
                                                                 replace=False)
                training_observations = self.observations[~self.observations.index.isin(held_out_observations.index)]

                # Fit every model with a new set of random training data
                self.model_list = parallel(delayed(self._fit_job)
                    (m, training_observations, predictors,
                     loss_function=loss_function, method=method, 
                     optimizer_params=optimizer_params,
                     verbose=verbose, debug=debug) for m in self.model_list)
                
                # Predict with every model on a new set of random test data
                held_out_predictions = [model.predict(held_out_observations, predictors=predictors) for model in self.model_list]
                held_out_predictions = np.array(held_out_predictions).T
        
                # Special funtion for use inside scipy.optimize routines
                def weighted_loss(w):
                    w = np.array([w])
                    w = w/w.sum()
                    pred = (held_out_predictions * w).sum(1)
                    return loss(held_out_observations.doy.values, pred)
                
                # Find the weights of each model with minimize RMSE
                iteration_weights = utils.optimize.fit_parameters(function_to_minimize = weighted_loss,
                                                                  bounds = weight_bounds,
                                                                  method='DE',
                                                                  optimizer_params=optimizer_params,
                                                                  results_translator=translate_scipy_weights)
                iteration_weights = iteration_weights / iteration_weights.sum()
                self.fitted_weights[i] = iteration_weights
        
            self.weights = self.fitted_weights.mean(0)
            
            # Sum of weights should equal or be extremely close to 1
            summed_weights = self.weights.sum().round(5)
            if summed_weights != 1.0:
                raise RuntimeError('Weights do not sum to 1, got '+str(summed_weights))
            
            # Refit the core models one last time to the full dataset. These 
            # fitted models will be compbined with the weights for predictions
            self.model_list = parallel(delayed(self._fit_job)
                (m, observations, predictors,
                 loss_function=loss_function, method=method, 
                 optimizer_params=optimizer_params,
                 verbose=verbose, debug=debug) for m in self.model_list)

    def predict(self, to_predict=None, predictors=None,
                aggregation = 'mean', n_jobs=1, **kwargs):
        """Make predictions..

        Predictions will be made using each core models, then a final average
        model derrived using the fitted weights.

        Parameters:
            see core model description
            
            aggregation : str
                Either 'weighted_mean' to get a normal prediciton, or 'none'
                to get predictions for all models. If using 'none' this returns
                a tuple of (weights, predictions). 
            
            n_jobs : int
                number of parallel processes to use

        """

        self._check_parameter_completeness()
        
        if not isinstance(aggregation, str):
            raise TypeError('aggregation should be a string. got: ' + str(type(aggregation)))
            
        if predictors is None and to_predict is None:
            predictors = self.predictors
            to_predict = self.observations

        predictions = Parallel(n_jobs = n_jobs)(delayed(self._predict_job)
            (m, to_predict = to_predict, predictors = predictors,
             aggregation=aggregation, **kwargs)
            for m in self.model_list)

        predictions = np.array(predictions)

        if aggregation=='mean':
            # Transpose to align the model axis with the 1D weight array
            # then transpose back to sum the weighted predictions.
            return (predictions.T * self.weights).T.sum(0)
        elif aggregation=='none':
            return self.weights, predictions
        else:
            raise ValueError('Unknown aggregation: ' + str(aggregation))

    def _check_parameter_completeness(self):
        """Make sure all parameters have been set from fitting or loading at initialization"""
        if not len(self.weights) == len(self.model_list):
            raise RuntimeError('Model not fit')
        
        [m._check_parameter_completeness() for m in self.model_list]
    
    def get_params(self):
        self._check_parameter_completeness()

        core_model_info = []
        for i, model in enumerate(self.model_list):
            core_model_info.append(deepcopy(model._get_model_info()))
            core_model_info[-1].update({'weight': self.weights[i]})
        return core_model_info

    def _get_model_info(self):
        # essentially the same as get_params() but is in a format capable of
        # being loaded again later by _parse_fully_fitted_model()
        return {'model_name': type(self).__name__,
                'core_models': self.get_params()}

    def get_weights(self):
        self._check_parameter_completeness()
        return self.weights
    
