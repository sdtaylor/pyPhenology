import numpy as np
from scipy import optimize


def get_loss_function(method):
    if method == 'rmse':
        return lambda obs, pred: np.sqrt(np.mean((obs - pred)**2))
    elif method == 'aic':
        return lambda obs, pred, n_param: len(obs) * np.log(np.mean((obs - pred)**2)) + 2 * (n_param + 1)
    else:
        raise ValueError('Unknown loss method: ' + method)


def validate_optimizer_parameters(optimizer_method, optimizer_params):
    sensible_defaults = {'DE': {'testing': {'maxiter': 5,
                                            'popsize': 10,
                                            'mutation': (0.5, 1),
                                            'recombination': 0.25,
                                            'disp': False},
                                'practical': {'maxiter': 1000,
                                              'popsize': 50,
                                              'mutation': (0.5, 1),
                                              'recombination': 0.25,
                                              'disp': False},
                                'intensive': {'maxiter': 10000,
                                              'popsize': 100,
                                              'mutation': (0.1, 1),
                                              'recombination': 0.25,
                                              'disp': False},
                                },
                         'BF': {'testing': {'Ns': 2,
                                            'finish': optimize.fmin_bfgs,
                                            'disp': False},
                                'practical': {'Ns': 20,
                                              'finish': optimize.fmin_bfgs,
                                              'disp': False},
                                'intensive': {'Ns': 40,
                                              'finish': optimize.fmin_bfgs,
                                              'disp': False}},
                         'BH': {'testing': {'niter': 100,
                                            'T': 0.5,
                                            'stepsize': 0.5,
                                            'disp': False},
                                'practical': {'niter': 50000,
                                              'T': 0.5,
                                              'stepsize': 0.5,
                                              'disp': False},
                                'intensive': {'niter': 500000,
                                              'T': 0.5,
                                              'stepsize': 0.5,
                                              'disp': False}}
                         }

    if isinstance(optimizer_params, str):
        try:
            optimizer_params = sensible_defaults[optimizer_method][optimizer_params]
        except KeyError:
            raise ValueError('Unknown sensible parameter string: ' + optimizer_params)

    elif isinstance(optimizer_params, dict):
        pass
    else:
        raise TypeError('Invalid optimizer parameters. Must be str or dictionary')

    return optimizer_params


def fit_parameters(function_to_minimize, bounds, method, results_translator,
                   optimizer_params, verbose=False):
    """Internal functions to estimate model parameters. 

    Methods
    -------
    'DE', Differential evolution
        Uses a large number of randomly specified parameters which converge
        on a global optimum. 

        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html

    'BF', Brute force
        Searches for the best parameter set within a confined space. Can take
        an extremely long time if used beyond 2 or 3 parameters.

        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brute.html

    'SA', Simulated annealing
        The most commonly used method in phenology modeling. Not yet implemented
        here as scipy has discontinued it in favor of basin hopping.

        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.anneal.html

    'BH, Basin hopping
        Starts off in a search space randomly, "hopping" around until a suitable
        minimum value is found.

        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html

    Parameters
    ----------
    funtions_to_minimize : func
        A minimizer function to pass to the optimizer model. Normally
        models._base_model.scipy_error

    bounds : list
        List of tuples specifying the upper and lower search space,
        where each tuple represents a model parameter

    method : str
        Optimization method to use

    results_translator : func
        A function to translate the optimizer output to a dictionary

    optimzier_parms : dict
        parameters to pass to the scipy optimizer

    Returns
    -------
    fitted_parameters : dict
        Dictionary of fitted parameters

    """
    if not isinstance(method, str):
        raise TypeError('method should be string, got ' + type(method))

    if method == 'DE':
        optimizer_params = validate_optimizer_parameters(optimizer_method=method,
                                                         optimizer_params=optimizer_params)

        optimize_output = optimize.differential_evolution(function_to_minimize,
                                                          bounds=bounds,
                                                          **optimizer_params)
        fitted_parameters = results_translator(optimize_output['x'])

    elif method == 'BH':
        optimizer_params = validate_optimizer_parameters(optimizer_method=method,
                                                         optimizer_params=optimizer_params)
        # optimize.bashinhopping takes an initial guess value, so here
        # choose one randomly from the (low,high) search ranges given
        initial_guess = [float(np.random.randint(l, h)) for l, h in bounds]

        optimize_output = optimize.basinhopping(function_to_minimize,
                                                x0=initial_guess,
                                                **optimizer_params,
                                                minimizer_kwargs={'method': 'L-BFGS-B',
                                                                  'bounds': bounds})
        fitted_parameters = results_translator(optimize_output['x'])

    elif method == 'SE':
        raise NotImplementedError('Simulated Annealing not working yet')
    elif method == 'BF':
        optimizer_params = validate_optimizer_parameters(optimizer_method=method,
                                                         optimizer_params=optimizer_params)

        # BF takes a tuple of tuples instead of a list of tuples like DE
        bounds = tuple(bounds)

        optimize_output = optimize.brute(func=function_to_minimize,
                                         ranges=bounds,
                                         **optimizer_params)

        fitted_parameters = results_translator(optimize_output)
    else:
        raise ValueError('Uknown optimizer method: ' + str(method))

    if verbose:
        print('Optimizer method: {x}\n'.format(x=method))
        print('Optimizer parameters: \n {x}\n'.format(x=optimizer_params))

    return fitted_parameters
