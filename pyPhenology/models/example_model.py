from . import utils
from .base import BaseModel


class Model(BaseModel):
    """Example to show model API"""

    def __init__(self, parameters={}):
        raise NotImplementedError('This model is for documentation only')
        BaseModel.__init__(self)
        self.all_required_parameters = {}
        self._organize_parameters(parameters)

    def _apply_model(self, temperature, doy_series, t1, T, F):
        pass
