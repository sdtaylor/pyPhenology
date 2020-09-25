from pyPhenology import utils
import pytest

def test_invalid_model_name():
    with pytest.raises(TypeError):
        utils.load_model(123)
        
def test_unknown_model_name():
    with pytest.raises(ValueError):
        utils.load_model('asdf')

def test_invalid_saved_model_type():
    with pytest.raises(TypeError):
        utils.load_saved_model(123)