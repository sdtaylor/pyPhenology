from pyPhenology import utils
import pytest

@pytest.mark.parametrize('dataset', [('vaccinium'),('aspen')])
def test_dataset_names(dataset):
    obs, predictors = utils.load_test_data(dataset)
    assert len(obs) > 0

@pytest.mark.parametrize('dataset', [('asdf'),('aspennn')])
def test_unknown_dataset_name(dataset):
    with pytest.raises(ValueError):
        obs, predictors = utils.load_test_data(name = dataset)
    
@pytest.mark.parametrize('dataset', [(123),(False),(['aspen']),(lambda x:x)])
def test_invalid_dataset_name(dataset):
    'dataset name needs to be a string'
    with pytest.raises(TypeError):
        obs, predictors = utils.load_test_data(name = dataset)
    
@pytest.mark.parametrize('phenophase', [('budburst'),('flowers'),('colored_leaves'),('all'),
                                        (371),(501),(498)])
def test_dataset_phenophase(phenophase):
    obs, predictors = utils.load_test_data(name='aspen', phenophase=phenophase)
    assert len(obs) > 0
    
@pytest.mark.parametrize('phenophase', [('asdf'),(123)])
def test_unknown_dataset_phenophase(phenophase):
    with pytest.raises(ValueError):
        obs, predictors = utils.load_test_data(name='aspen', phenophase=phenophase)

@pytest.mark.parametrize('phenophase', [(None),([371]),(lambda x:x)])
def test_invalid_dataset_phenophase(phenophase):
    with pytest.raises(TypeError):
        obs, predictors = utils.load_test_data(name='aspen', phenophase=phenophase)
        
