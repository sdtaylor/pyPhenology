import numpy as np
from . import utils
from ..utils import load_model
from . import validation
from copy import deepcopy
import warnings


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
                Also if it is a saved model file then core_model and 
                num_bootstrap are ignored and inferred from the file.        
        """
        self.observations = None
        self.predictors = None
        self.model_list = []
        if isinstance(parameters, str):
            # A filename pointing toward a file from save_params()
            # core_model and num_bootstraps is ignored
            model_info = utils.misc.read_saved_model(model_file=parameters)

            if type(self).__name__ != model_info['model_name']:
                raise RuntimeError('Saved model file does not match model class. ' +
                                   'Saved file is {a}, this model is {b}'.format(a=model_info['model_name'],
                                                                                 b=type(self).__name__))

            for bootstrap_iteration in model_info['parameters']:
                Model = load_model(bootstrap_iteration['model_name'])
                self.model_list.append(Model(parameters=bootstrap_iteration['parameters']))

        elif isinstance(parameters, dict):
            # Custom parameter values to pass to each bootstrap model
            if core_model is None or num_bootstraps is None:
                raise TypeError('core_model and num_bootstraps must be set')

            validation.validate_model(core_model())
            for i in range(num_bootstraps):
                self.model_list.append(core_model(parameters=parameters))

        elif isinstance(parameters, list):
            # If its the output of BootstrapModel.get_params()
            for bootstrap_iteration in parameters:
                bootstrap_iteration.pop('bootstrap_num')
                self.model_list.append(core_model(parameters=bootstrap_iteration))
        else:
            raise TypeError('parameters must be str or dict, got: ' + str(type(parameters)))

    def fit(self, observations, predictors, **kwargs):
        """Fit the underlying core models

        Parameters:
            observations : dataframe
                pandas dataframe of phenology observations

            predictors : dataframe
                pandas dataframe of associated predictorss

            kwargs :
                Other arguments passed to core model fitting (eg. optimzer methods)
        """
        # TODO: do the predictors transform here cause so it doesn't get reapated a bunch
        # need to wait till fit takes arrays directly
        self.observations = observations
        self.predictors = predictors

        for model in self.model_list:
            obs_shuffled = observations.sample(frac=1, replace=True).copy()
            model.fit(obs_shuffled, predictors, **kwargs)

    def predict(self, to_predict=None, predictors=None, aggregation='mean', **kwargs):
        """Make predictions from the bootstrapped models.

        Predictions will be made using each of the bootstrapped models.
        The final results will be the mean or median of all bootstraps, or all bootstrapped
        model results in 2d array.

        Parameters:
            aggregation : str
                Either 'mean','median', or 'none'. 'none' return *all* predictions
                in an array of size (num_bootstraps, num_samples)

        """
        # Get the organized predictors. This is done from the 1st model in the
        # list, but since they're all the same model it should be valid for all.
        # Only works if there are new predictors. Otherwise the data used for
        # fitting will be  used.
        # TODO: impliment this
        # if predictors is not None:
        #   predictors = self.model_list[0]._organize_predictors(observations=to_predict,
        #                                                         predictors=predictors,
        #                                                         for_prediction=True)
        if predictors is None:
            predictors = self.predictors
        if to_predict is None:
            to_predict = self.observations

        predictions = []
        for model in self.model_list:
            predictions.append(model.predict(to_predict=to_predict,
                                             predictors=predictors,
                                             **kwargs))

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

    def get_params(self):

        all_params = []
        for i, model in enumerate(self.model_list):
            all_params.append(deepcopy(model.get_params()))
            all_params[-1].update({'bootstrap_num': i})

        return all_params

    def save_params(self, filename, overwrite=False):
        """Save model parameters

        Note this will save details on all bootstrapped models, and 
        can only be loaded again as a bootstrap model.

        Parameters:
            filename : str
                Filename to save model to. Note this can be loaded again by
                passing the filename in the ``parameters`` argument, but only
                with the BootstrapModel.

            overwrite : bool
                Overwrite the file if it exists

        """
        model_parameter_status = [m._parameters_are_set() for m in self.model_list]
        if not np.all(model_parameter_status):
            raise RuntimeError('Cannot save bootstrap model, not all parameters set')

        core_model_info = []
        for i, model in enumerate(self.model_list):
            core_model_info.append(deepcopy(model._get_model_info()))
            core_model_info[-1].update({'bootstrap_num': i})

        model_info = {'model_name': type(self).__name__,
                      'parameters': core_model_info}

        utils.misc.write_saved_model(model_info=model_info,
                                     model_file=filename,
                                     overwrite=overwrite)
