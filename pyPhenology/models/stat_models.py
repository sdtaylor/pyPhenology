from . import utils
from .base import BaseModel
import warnings


class Linear(BaseModel):
    """Linear Regression Model

    A 2 parameter regression model with :math:`DOY` as
    the response variable. 

    .. math::
        DOY = \\beta_{1} + \\beta_{2}T_{mean}

    where :math:`T_{mean}` is the mean temperature of the time period specified.
    By default it is set to model spring phenology (with mean temperature of
    Jan. 1 - March 31). This can also be used for fall phenology by setting the
    time parameters to the late summer/fall season.

    Parameters:
        intercept : int | float
            | :math:`\\beta_{1}`, intercept of the model
            | default : (-67,298)

        slope : int | float
            | :math:`\\beta_{1}`, Slope of the model
            | default : (-25,25)

        time_start : int
            | Start doy for the season of interest
            | default : 1 (Jan 1)

        time_length : int
            | The length of the season in days
            | default : 90

    """

    def __init__(self, parameters={}):
        BaseModel.__init__(self)
        self.all_required_parameters = {'intercept': (-67, 298), 'slope': (-25, 25),
                                        'time_start': 0, 'time_length': 90}
        self._organize_parameters(parameters)
        self._required_data = {'predictor_columns': ['site_id', 'year', 'doy', 'temperature'],
                               'predictors': ['temperature', 'doy_series']}

    def _apply_model(self, temperature, doy_series, intercept, slope,
                     time_start, time_length):
        mean_spring_temp = utils.transforms.mean_temperature(temperature, doy_series,
                                                             start_doy=time_start,
                                                             end_doy=time_start + time_length)
        return mean_spring_temp * slope + intercept


class Naive(BaseModel):
    """A naive model of the spatially interpolated mean

    This is the mean doy for an event adjusted for latitude, essentially    
    a 2 parameter regression model with :math:`DOY` as
    the response variable. 

    .. math::
        DOY = \\beta_{1} + \\beta_{2}Latitude

    This model requires only a latitude column in the predictors
    for each unique site_id

    Parameters:
        intercept : int | float
            | :math:`\\beta_{1}`, intercept of the model
            | default : (-67,298)

        slope : int | float
            | :math:`\\beta_{1}`, Slope of the model
            | default : (-25,25)

    """

    def __init__(self, parameters={}):
        BaseModel.__init__(self)
        self.all_required_parameters = {'intercept': (-67, 298), 'slope': (-25, 25)}
        self._organize_parameters(parameters)
        self._required_data = {'predictor_columns': ['site_id', 'year', 'doy'],
                               'predictors': ['latitude']}

    def _organize_predictors(self, predictors, observations, for_prediction):
        site_latitudes = predictors[['site_id', 'latitude']].drop_duplicates()

        # Check to make sure a site has only one latitude assigned to it
        duplicate_sites = site_latitudes['site_id'].duplicated()
        if duplicate_sites.any():
            duplicate_site_ids = site_latitudes['site_id'][duplicate_sites].drop_duplicates().tolist()
            warnings.warn('sites with >1 latitude, keeping the first occurance.\n ' +
                          'duplicate sites: ' + str(duplicate_site_ids))
            # Keep only the first instance of any duplicate site
            site_latitudes = site_latitudes[~duplicate_sites]

        obs_with_latitude = observations.merge(site_latitudes, on='site_id')

        if for_prediction:
            return {'latitude': obs_with_latitude.latitude.values}
        else:
            self.fitting_predictors = {'latitude': obs_with_latitude.latitude.values}
            self.obs_fitting = obs_with_latitude.doy.values

    def _validate_formatted_predictors(self, predictors):
        pass

    def _apply_model(self, latitude, intercept, slope):
        return intercept + (slope * latitude)
