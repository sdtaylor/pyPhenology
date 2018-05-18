import pandas as pd
import pkg_resources
from . import models
from warnings import warn


def load_test_data(name='vaccinium', phenophase='all'):
    """Pre-loaded phenology data

    Datasets are available with the package. They include multiple phenophases
    and associated daily mean temperature data derived from the PRISM 
    climate dataset. 

    Without any arguments it will return the vaccinium dataset for both
    phenophases.

    Available datasets:
        'vaccinium'
            Vaccinium corymbosum phenology from Harvard Forest
            Both flowers (phenophase 501) and budburst (phenophase 371)
        'aspen'
            Populus tremuloides (aspen) phenology from the National Phenology
            Dataset. Has flowers  (phenophase 501), budburst (phenophase 371),
            and colored leaves for fall senesence (phenophase 498)

    Parameters:
        name : str, optional
            Name of the test dataset

        phenophase : str | int, optional
            Name of the phenophase. Either 'budburst','flowers', 'colored_leaves',
            or 'all'. Or the phenophase id (371 or 501, or 498)

    Returns:
        obs, temp : tuple
            Pandas dataframes of phenology observations
            and associated temperatures.
    """

    if not isinstance(name, str):
        raise TypeError('Unknown name type. Expected str, got ' + type(name))

    if name == 'vaccinium':
        obs_file = 'data/vaccinium_obs.csv'
        temp_file = 'data/vaccinium_temperature.csv'
    elif name == 'aspen':
        obs_file = 'data/aspen_obs.csv'
        temp_file = 'data/aspen_temperature.csv.gz'
    else:
        raise ValueError('Uknown dataset name: ' + str(name))

    if isinstance(phenophase, int):
        if phenophase not in [371, 501, 498]:
            raise ValueError('uknown phenophase: ' + str(phenophase))
        phenophase_ids = [phenophase]
    elif isinstance(phenophase, str):
        if phenophase == 'budburst':
            phenophase_ids = [371]
        elif phenophase == 'flowers':
            phenophase_ids = [501]
        elif phenophase == 'colored_leaves':
            phenophase_ids = [498]
        elif phenophase == 'all':
            phenophase_ids = [371, 501, 498]
        else:
            raise ValueError('unknown phenophase: ' + phenophase)
    else:
        raise TypeError('unknown phenophase type. Expected str or int, got ' + str(type(phenophase)))

    obs_file = pkg_resources.resource_filename(__name__, obs_file)
    temp_file = pkg_resources.resource_filename(__name__, temp_file)
    obs = pd.read_csv(obs_file)
    temp = pd.read_csv(temp_file)

    obs = obs[obs.phenophase.isin(phenophase_ids)]
    
    if len(obs) == 0:
        raise RuntimeError('Phenophase {p} not available for dataset {d}'.format(p=phenophase,d=name))

    return obs, temp


def load_model(name):
    """Load a model via a string

    Options are ``['ThermalTime','Uniforc','Unichill','Alternating','MSB',
                   'Sequential','Linear','M1','FallCooling','Naive']``
    """
    if not isinstance(name, str):
        raise TypeError('name must be string, got' + type(name))
    if name == 'ThermalTime':
        return models.ThermalTime
    elif name == 'Uniforc':
        return models.Uniforc
    elif name == 'Unichill':
        return models.Unichill
    elif name == 'Alternating':
        return models.Alternating
    elif name == 'MSB':
        return models.MSB
    elif name == 'Sequential':
        return models.Sequential
    elif name == 'Linear':
        return models.Linear
    elif name == 'M1':
        return models.M1
    elif name == 'FallCooling':
        return models.FallCooling
    elif name == 'Naive':
        return models.Naive
    else:
        raise ValueError('Unknown model name: ' + name)


def load_saved_model(filename):
    """Load a previously saved model file

    Returns the model object with parameters preloaded.
    """
    if not isinstance(filename, str):
        raise TypeError('filename must be string, got' + type(filename))

    model_info = models.utils.misc.read_saved_model(filename)

    # These ensemble methods have their own code for loading saved files
    if model_info['model_name'] == 'BootstrapModel':
        model = models.BootstrapModel(parameters=filename)
    elif model_info['model_name'] == 'WeightedEnsemble':
        model = models.WeightedEnsemble(core_models=filename)
    else:
        # For all other ones just need to pass the parameters
        Model = load_model(model_info['model_name'])
        model = Model(parameters=model_info['parameters'])

    return model


def check_data(observations, predictors, drop_missing=True, for_prediction=False):
    """Make sure observation and predictors data.frames are
    valid before submitting them to models.
    If observations are missing predictors data, optionally return
    a dataframe with those observations dropped.
    """
    original_obs_columns = observations.columns.values

    predictors_pivoted = predictors.pivot_table(index=['site_id', 'year'], columns='doy', values='temperature').reset_index()

    # This first and last day of temperature data can causes NA issues because
    # of leap years.If thats the case try dropping them
    first_doy_has_na = predictors_pivoted.iloc[:, 2].isna().any()  # first day will always be col 2
    if first_doy_has_na:
        first_doy_column = predictors_pivoted.columns[2]
        predictors_pivoted.drop(first_doy_column, axis=1, inplace=True)
        warn("""Dropped temperature data for doy {d} due to missing data. Most likely from leap year mismatch""".format(d=first_doy_column))

    last_doy_index = predictors_pivoted.shape[1] - 1
    last_doy_has_na = predictors_pivoted.iloc[:, last_doy_index].isna().any()
    if last_doy_has_na:
        last_doy_column = predictors_pivoted.columns[-1]
        predictors_pivoted.drop(last_doy_column, axis=1, inplace=True)
        warn("""Dropped temperature data for doy {d} due to missing data. Most likely from leap year mismatch""".format(d=last_doy_column))


    observations_with_temp = observations.merge(predictors_pivoted, on=['site_id', 'year'], how='left')

    original_sample_size = len(observations_with_temp)
    rows_with_missing_data = observations_with_temp.isnull().any(axis=1)
    missing_info = observations_with_temp[['site_id', 'year']][rows_with_missing_data].drop_duplicates()
    print(len(missing_info))
    if len(missing_info) > 0 and drop_missing:
        observations_with_temp.dropna(axis=0, inplace=True)
        n_dropped = original_sample_size - len(observations_with_temp)
        print('Dropped {n0} of {n1} observations because of missing data'.format(n0=n_dropped, n1=original_sample_size))
        print('\n Missing data from: \n' + str(missing_info))
        return observations_with_temp[original_obs_columns], predictors
    elif len(missing_info) > 0:
        print('Missing predictors values detected')
        print('\n Missing data from: \n' + str(missing_info))
        return observations, predictors
    else:
        return observations, predictors
