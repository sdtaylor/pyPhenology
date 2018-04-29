from . import utils
from .base import BaseModel


class Sequential(BaseModel):
    """The sequential model

    This uses a triangular response for chilling, and growing degree days for forcing.

    Parameters:
        t0 : int
            The doy which chilling accumulation beings

        c_t_min: int | float
            Triangular response parameter. The minimum temperature where
            chilling accumulates.

        c_t_opt: int | float
            Triangular response parameter. The optimum temperature where
            chilling accumulates.

        c_t_max: int | float
            Triangular response parameter. The maximum temperature where
            chilling accumulates.

        C : int | float
            Total chilling units required.

        f_t : int | float
            The threshold above which warming forcing accumulates

        F : int, > 0
            The total forcing units required
    """

    def __init__(self, parameters={}):
        BaseModel.__init__(self)
        self.all_required_parameters = {'t0': (-67, 298), 'c_t_min': (-25, 10), 'c_t_opt': (-10, 10),
                                        'c_t_max': (0, 10), 'C': (0, 100), 'f_t': (0, 20), 'F': (0, 1000)}
        self._organize_parameters(parameters)
        self._required_data = {'predictor_columns': ['site_id', 'year', 'doy', 'temperature'],
                               'predictors': ['temperature', 'doy_series']}

    def _apply_model(self, temperature, doy_series, t0, c_t_min, c_t_opt, c_t_max,
                     C, f_t, F):

        # Enforce realistic values for the triangle response parameters
        bad_values = True if c_t_min >= c_t_opt else False
        bad_values = True if c_t_min >= c_t_max else False
        bad_values = True if c_t_opt >= c_t_max else False

        if bad_values:
            to_return = temperature[0].copy()
            to_return[:] = 9999
            return to_return

        chill_days = utils.transforms.triangle_response(temperature.copy(), t_min=c_t_min,
                                                        t_opt=c_t_opt, t_max=c_t_max)
        chill_days[doy_series < t0] = 0
        chill_days = utils.transforms.forcing_accumulator(chill_days)

        # Where adequate chill has not yet accumulated
        temperature[chill_days < C] = 0
        # Warming threshold temperature
        temperature[temperature < f_t] = 0

        accumulated_gdd = utils.transforms.forcing_accumulator(temperature)

        return utils.transforms.doy_estimator(forcing=accumulated_gdd,
                                              doy_series=doy_series,
                                              threshold=F)
