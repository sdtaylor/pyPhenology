from pyPhenology import utils, models
from warnings import warn
import sys
import pytest

from numpy.random import default_rng
# Make sure some known parameters are estimated correctly
# The number roughly match the estimates from the original model
# codebase in github.com/sdtaylor/phenology_dataset_study.
# They don't match exactly because that original stuff uses DE
# while I'm using BF here for testing sake.

##############################
# Use a specific species dataset that should never change
leaves_obs, vaccinium_temp = utils.load_test_data(name='vaccinium', phenophase=371)
flowers_obs, vaccinium_temp = utils.load_test_data(name='vaccinium', phenophase=501)

aspen_leaves, aspen_temp = utils.load_test_data(name='aspen', phenophase=371)

# thorough but still relatively quick
thorough_DE_optimization = {'method':'DE', 'debug':True,
                            'optimizer_params':{'popsize':20,
                                                'maxiter':100,
                                                'mutation':1.5,
                                                'recombination':0.25}}


#######################################
# Setup test cases

test_cases=[]
test_cases.append({'test_name' : 'Thermal Time Vaccinium Leaves',
                   'model' : models.ThermalTime,
                   'fitting_obs':leaves_obs,
                   'fitting_temp':vaccinium_temp,
                   'expected_params':{'F':272},
                   'fitting_ranges':{'t1':0, 'T':0, 'F':(0,1000)},
                   'fitting_params':thorough_DE_optimization})

test_cases.append({'test_name' : 'Thermal Time Vaccinium Flowers',
                   'model' : models.ThermalTime,
                   'fitting_obs':flowers_obs,
                   'fitting_temp':vaccinium_temp,
                   'expected_params':{'F':448},
                   'fitting_ranges':{'t1':0, 'T':0, 'F':(0,1000)},
                   'fitting_params':thorough_DE_optimization})
                   
test_cases.append({'test_name' : 'Alternating Vaccinium Leaves',
                   'model' : models.Alternating,
                   'fitting_obs':leaves_obs,
                   'fitting_temp':vaccinium_temp,
                   'expected_params': {'a':618, 'b':-135, 'c':0},
                   'fitting_ranges':{'a':(600,700), 'b':(-200,-100), 'c':(0.009,0.02), 't1':0, 'threshold':5},
                   'fitting_params':thorough_DE_optimization})

test_cases.append({'test_name' : 'Alternating Vaccinium Flowers',
                   'model' : models.Alternating,
                   'fitting_obs':flowers_obs,
                   'fitting_temp':vaccinium_temp,
                   'expected_params': {'a':1010, 'b':-463, 'c':0},
                   'fitting_ranges':{'a':(1000,1100), 'b':(-500,-400), 'c':(0.001,0.01), 't1':0, 'threshold':5},
                   'fitting_params':thorough_DE_optimization})

test_cases.append({'test_name' : 'Uniforc Vaccinium Leaves',
                   'model' : models.Uniforc,
                   'fitting_obs':leaves_obs,
                   'fitting_temp':vaccinium_temp,
                   'expected_params': {'t1':85, 'b':0, 'c':11, 'F':5},
                   'fitting_ranges':{'t1':(50,100), 'b':(-5,5), 'c':(0,20), 'F':(0,20)},
                  'fitting_params':thorough_DE_optimization})
    
test_cases.append({'test_name' : 'Uniforc Vaccinium Flowers',
                   'model' : models.Uniforc,
                   'fitting_obs':flowers_obs,
                   'fitting_temp':vaccinium_temp,
                   'expected_params': {'t1':42, 'b':0, 'c':10, 'F':18},
                   'fitting_ranges':{'t1':(25,50), 'b':(-5,5), 'c':(0,20), 'F':(10,30)},
                   'fitting_params':thorough_DE_optimization})

test_cases.append({'test_name' : 'Unichill Vaccinium Flowers',
                   'model' : models.Unichill,
                   'fitting_obs':flowers_obs,
                   'fitting_temp':vaccinium_temp,
                   'expected_params': {'t0':-30,'C':81,'F':17,
                                          'b_f':0,'c_f':13,
                                          'a_c':0,'b_c':-14,'c_c':-21},
                   'fitting_ranges':{'t0':(-40,-10),'C':(50,100),'F':(5,30),
                                     'b_f':(-10,10),'c_f':(0,20),
                                     'a_c':(-10,10),'b_c':(-20,0),'c_c':(-25,-5)},
                   'fitting_params':thorough_DE_optimization})

test_cases.append({'test_name' : 'Linear Model Aspen leaves',
                   'model' : models.Linear,
                   'fitting_obs':aspen_leaves,
                   'fitting_temp':aspen_temp,
                   'expected_params':{'intercept': 118, 'slope': 0, 'time_start': 0, 'time_length': 90},
                   'fitting_ranges':{},
                   'fitting_params':thorough_DE_optimization})

test_cases.append({'test_name' : 'Naive Model Aspen leaves',
                   'model' : models.Naive,
                   'fitting_obs':aspen_leaves,
                   'fitting_temp':aspen_temp,
                   'expected_params':{'intercept': 156, 'slope': 0},
                   'fitting_ranges':{},
                   'fitting_params':thorough_DE_optimization})

#######################################
# Get estimates for all models
for case in test_cases:
    # default random state for each
    case['fitting_params']['optimizer_params']['seed'] = default_rng(1)
        
    model = case['model'](parameters = case['fitting_ranges'])
    model.fit(observations = case['fitting_obs'], predictors=case['fitting_temp'], 
              **case['fitting_params'])
    case['estimated_params'] = model.get_params()
    
########################################
# Setup tuples for pytest.mark.parametrize
test_cases = [(c['test_name'], c['expected_params'],c['estimated_params']) for c in test_cases]

@pytest.mark.parametrize('test_name, expected_params, estimated_params', test_cases)
def test_know_parameter_values(test_name, expected_params, estimated_params):
    all_values_match = True
    # Values are compared as ints since precision past that is not the goal here.
    for param, expected_value in expected_params.items():
        if int(estimated_params[param]) != expected_value:
            all_values_match = False
    
    # Slight variations can potentially happen with updates in the algorithms of
    # of underlying scipy/numpy packages. They are likely not showstoppers, so just
    # use warnings if things do not match.
    if not all_values_match:
        warn('Not all values match: {n} \n' \
             'Expected: {e} \n Got: {g}'.format(n=test_name,
                                                e=expected_params,
                                                g=estimated_params))

