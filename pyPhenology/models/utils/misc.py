import json
import os
from warnings import warn


def temperature_only_data_prep(observations, predictors, for_prediction=False,
                               verbose=True):
    """Create a numpy array of shape (a,b), where b
    is equal to the sample size in observations, and a is
    equal to the number of days in the yearly time
    series of temperature (ie. Jan 1 - July 30).
    Using a numpy array in this way allows for very 
    efficient processing of phenology models.

    Parameters
    ----------
    observations : Pandas Dataframe
        A data frame with columns ['doy','year','site_id'],
        where every row is an observation for an observed
        phenological event.

    predictors : Pandas Dataframe
        A Dataframe with columns['temperature','year','site_id']
        which matches to the sites and years in observations.

    for_prediction : bool
        Do not return observed_doy, or expect a doy column in observations.

    verbose : bool
        Show details of processing

    Returns
    -------
    observed_doy : Numpy array
        a 1D array of the doy of each observation

    temperature_array : Numpy array
        a 2D array described above

    doy_series : Numpy array
        1D array with length equal to the number of columns
        in temperature_array. Represents the doy values.
        (ie. doy 0 = Jan 1)

    """
    predictors = predictors[['doy', 'site_id', 'year', 'temperature']].copy()
    doy_series = predictors.doy.dropna().unique()
    doy_series.sort()
    predictors = predictors.pivot_table(index=['site_id', 'year'], columns='doy', values='temperature').reset_index()

    # This first and last day of temperature data can causes NA issues because
    # of leap years.If thats the case try dropping them
    first_doy_has_na = predictors.iloc[:, 2].isna().any()  # first day will always be col 2
    if first_doy_has_na:
        first_doy_column = predictors.columns[2]
        predictors.drop(first_doy_column, axis=1, inplace=True)
        doy_series = doy_series[1:]
        warn("""Dropped temperature data for doy {d} due to missing data. Most likely from leap year mismatch""".format(d=first_doy_column))

    last_doy_index = predictors.shape[1] - 1
    last_doy_has_na = predictors.iloc[:, last_doy_index].isna().any()
    if last_doy_has_na:
        last_doy_column = predictors.columns[-1]
        predictors.drop(last_doy_column, axis=1, inplace=True)
        doy_series = doy_series[:-1]
        warn("""Dropped temperature data for doy {d} due to missing data. Most likely from leap year mismatch""".format(d=last_doy_column))


    # Give each observation a temperature time series
    obs_with_temp = observations.merge(predictors, on=['site_id', 'year'], how='left')

    # Deal with any site/years that don't have temperature data
    original_sample_size = len(obs_with_temp)
    rows_with_missing_data = obs_with_temp.isnull().any(axis=1)
    missing_info = obs_with_temp[['site_id', 'year']][rows_with_missing_data].drop_duplicates()
    if len(missing_info) > 0:
        obs_with_temp.dropna(axis=0, inplace=True)
        n_dropped = original_sample_size - len(obs_with_temp)
        warn('Dropped {n0} of {n1} observations because of missing data'.format(n0=n_dropped, n1=original_sample_size) +
             '\n Missing data from: \n' + str(missing_info))

    observed_doy = obs_with_temp.doy.values
    temperature_array = obs_with_temp[doy_series].values.T

    if for_prediction:
        return temperature_array, doy_series
    else:
        return observed_doy, temperature_array, doy_series


def read_saved_model(model_file):
    with open(model_file, 'r') as f:
        m = json.load(f)
    return m


def write_saved_model(model_info, model_file, overwrite):
    if os.path.exists(model_file) and not overwrite:
        raise RuntimeWarning('File {f} exists. User overwrite=True to overwite'.format(f=model_file))
    else:
        with open(model_file, 'w') as f:
            json.dump(model_info, f, indent=4)
