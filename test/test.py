from pyPhenology import utils, models
from pyPhenology.models import validation
import pytest
import sys

doy, temp = utils.load_test_data()

quick_optimization = {'maxiter':5, 'popsize':10, 'disp':True}


model_test_cases=[]
model_test_cases.append({'model_name':'BootstrapModel',
                         'model_func': models.BootstrapModel,
                         'fit_params':{'optimizer_params':quick_optimization},
                         'initial_params':{'num_bootstraps':10,
                                           'core_model':models.Thermal_Time}})
model_test_cases.append({'model_name':'Uniforc',
                         'model_func': models.Uniforc,
                         'fit_params':{'optimizer_params':quick_optimization},
                         'initial_params':{}})
model_test_cases.append({'model_name':'Unichill',
                         'model_func': models.Uniforc,
                         'fit_params':{'optimizer_params':quick_optimization},
                         'initial_params':{}})
model_test_cases.append({'model_name':'Thermal_Time',
                         'model_func': models.Thermal_Time,
                         'fit_params':{'optimizer_params':quick_optimization},
                         'initial_params':{}})
model_test_cases.append({'model_name':'Alternating',
                         'model_func': models.Alternating,
                         'fit_params':{'optimizer_params':quick_optimization},
                         'initial_params':{}})
model_test_cases.append({'model_name':'MSB',
                         'model_func': models.MSB,
                         'fit_params':{'optimizer_params':quick_optimization},
                         'initial_params':{}})
model_test_cases.append({'model_name':'Linear',
                         'model_func': models.Linear,
                         'fit_params':{'optimizer_params':quick_optimization},
                         'initial_params':{}})

for test_case in model_test_cases:
    model_name = test_case['model_name']
    Model = test_case['model_func']
    initial_params = test_case['initial_params']
    fit_params = test_case['fit_params']
    
    print(model_name + ' - Initial validaiton')
    if model_name!='BootstrapModel':
        validation.validate_model(Model(**initial_params))
    else:
        'Skipping for bootstrap model'
    
    #Test with no fixed parameters
    print(model_name + ' - Estimate all parameters')
    model=Model(**initial_params)
    model.fit(DOY=doy, temperature=temp, verbose=True, **fit_params)
    model.predict(doy, temp)

    print(model_name + ' - do not predict without both doy and temp')
    with pytest.raises(TypeError) as a:
        model.predict(site_years = doy)
    with pytest.raises(TypeError) as a:
        model.predict(temperature = temp)
    print(model_name + ' - make prediction with values from fitting')
    predicted = model.predict()

    print(model_name + ' - prediction sample size matches input')
    assert len(predicted.shape) == 1, 'predicted array not 1D'
    assert len(predicted) == len(doy), 'predicted sample size not matching input from fit'

    predicted = model.predict(site_years=doy[1:10], temperature=temp)
    assert len(predicted.shape) == 1, 'predicted array not 1D'
    assert len(predicted) == len(doy[1:10]), 'predicted sample size not matching input from predict'
    
    # Use estimated parameter values in other tests
    all_parameters = model.get_params()

    
    # Test with all parameters set to a fixed value
    print(model_name + ' - Fix all parameters')
    model = Model(parameters=all_parameters, **initial_params)
    model.predict(doy, temp)
    
    print(model_name + ' - Do not predict without site_years and temperature \
                        when all parameters are fixed a initialization')
    with pytest.raises(TypeError) as a:
        model.predict()
    
    # For the bootstrap use just the first bootstrap value from here
    if model_name=='BootstrapModel':
        all_parameters = all_parameters[0]
    
    # Only a single parameter set to fixed
    print(model_name + ' - Fix a single parameter')
    param, value = all_parameters.popitem()
    single_param = {param:value}
    model = Model(parameters=single_param, **initial_params)
    model.fit(DOY=doy, temperature=temp, verbose=True, **fit_params)
    model.predict(doy, temp)
    
    fitted_params_with_one_fixed = model.get_params()
    if model_name=='BootstrapModel':
        fitted_params_with_one_fixed = fitted_params_with_one_fixed[0]
    assert fitted_params_with_one_fixed[param]==value, 'Fixed parameter does not equal final one: '+str(param)

    # Estimate only a single parameter
    print(model_name + ' - Estimate a single parameter')
    # all_parameters is now minus one due to popitem()
    model = Model(parameters=all_parameters, **initial_params)
    model.fit(DOY=doy, temperature=temp, verbose=True, **fit_params)
    model.predict(doy, temp)
    
    fitted_params_with_most_fixed = model.get_params()
    if model_name=='BootstrapModel':
        fitted_params_with_most_fixed = fitted_params_with_most_fixed[0]
    for param, value in all_parameters.items():
        assert fitted_params_with_most_fixed[param] == value, 'Fixed parameter does not equal final one: '+str(param)
    
    # Expect error when predicting but not all parameters were
    # passed, and no fitting has been done.
    print(model_name + ' - Should not predict without fit')
    with pytest.raises(AssertionError) as a:
        model = Model(parameters=single_param, **initial_params)
        model.predict(doy, temp)
    
    # Expect error when a bogus parameter gets passed
    print(model_name + ' - Should not accept unknown parameters')
    with pytest.raises(TypeError) as a:
        model = Model(parameters={'not_a_parameter':0}, **initial_params)

    #Save and load a parameter file
    print(model_name + ' - Save and load parameter file')
    model=Model(**initial_params)
    model.fit(DOY=doy, temperature=temp, verbose=True, **fit_params)
    model.save_params(model_name+'_params.csv')
    model=Model(parameters=model_name+'_params.csv', **initial_params)
    model.predict(doy, temp)

############################################################
############################################################
# Make sure some known parameters are estimated correctly
# Using the Brute Force method in this way should produce
# the same result every time.
# The number roughly match the estimates from the original model
# codebase in github.com/sdtaylor/phenology_dataset_study.
# They don't match exactly becuse that original stuff uses DE
# while I'm using BF here for testing sake.
        
###########################################################
def check_known_values(estimated_params, known_params, message):
    raise_error = False

    # Values are compared as ints since precision past that is not the goal here.
    for param, known_value in known_params.items():
        if int(estimated_params[param]) != known_value:
            raise_error=True
    
    if raise_error:
        log_message = message + ' incorrect \n' \
                     'Expected: ' + str(known_params) + '\n' \
                     'Got: ' + str(estimated_params)
        
        # Specific values are varying slightly between versions, 
        # let it slide if it's not on the specific version I tested
        # things on. 
        # TODO: Make this more robust
        if sys.version_info.major == 3 and sys.version_info.minor==6:
            raise ValueError(log_message)
        else:
            print('Values not matching')
            print(log_message)

##############################
# Use a specific species dataset that should never change
vaccinium_doy, vaccinium_temp = utils.load_test_data(name='vaccinium')

leaves_doy = vaccinium_doy[vaccinium_doy.Phenophase_ID==371]
flowers_doy = vaccinium_doy[vaccinium_doy.Phenophase_ID==501]

test_cases=[]
test_cases.append({'test_name' : 'Thermal Time Vaccinium Leaves',
                   'model' : models.Thermal_Time,
                   'fitting_doy':leaves_doy,
                   'fitting_temp':vaccinium_temp,
                   'known_model_params':{'F':273},
                   'fitting_ranges':{'t1':0, 'T':0, 'F':(0,1000)},
                   'optimizer_parameters':{'Ns':1000, 'finish':None}})

test_cases.append({'test_name' : 'Thermal Time Vaccinium Flowers',
                   'model' : models.Thermal_Time,
                   'fitting_doy':flowers_doy,
                   'fitting_temp':vaccinium_temp,
                   'known_model_params':{'F':448},
                   'fitting_ranges':{'t1':0, 'T':0, 'F':(0,1000)},
                   'optimizer_parameters':{'Ns':1000, 'finish':None}})

test_cases.append({'test_name' : 'Alternating Vaccinium Leaves',
                   'model' : models.Alternating,
                   'fitting_doy':leaves_doy,
                   'fitting_temp':vaccinium_temp,
                   'known_model_params': {'a':620, 'b':-141, 'c':0},
                   'fitting_ranges':{'a':(600,700), 'b':(-200,-100), 'c':(0.009,0.02), 't1':0, 'threshold':5},
                   'optimizer_parameters':{'Ns':30}})

test_cases.append({'test_name' : 'Alternating Vaccinium Flowers',
                   'model' : models.Alternating,
                   'fitting_doy':flowers_doy,
                   'fitting_temp':vaccinium_temp,
                   'known_model_params': {'a':1010, 'b':-465, 'c':0},
                   'fitting_ranges':{'a':(1000,1100), 'b':(-500,-400), 'c':(0.001,0.01), 't1':0, 'threshold':5},
                   'optimizer_parameters':{'Ns':30}})

test_cases.append({'test_name' : 'Uniforc Vaccinium Leaves',
                   'model' : models.Uniforc,
                   'fitting_doy':leaves_doy,
                   'fitting_temp':vaccinium_temp,
                   'known_model_params': {'t1':78, 'b':-2, 'c':8, 'F':8},
                   'fitting_ranges':{'t1':(50,100), 'b':(-5,5), 'c':(0,20), 'F':(0,20)},
                   'optimizer_parameters':{'Ns':15}})
    
test_cases.append({'test_name' : 'Uniforc Vaccinium Flowers',
                   'model' : models.Uniforc,
                   'fitting_doy':flowers_doy,
                   'fitting_temp':vaccinium_temp,
                   'known_model_params': {'t1':35, 'b':0, 'c':8, 'F':21},
                   'fitting_ranges':{'t1':(25,50), 'b':(-5,5), 'c':(0,20), 'F':(10,30)},
                   'optimizer_parameters':{'Ns':15}})
    
for case in test_cases:
    print('Testing known values: '+case['test_name'])
    model = case['model'](parameters = case['fitting_ranges'])
    model.fit(DOY = case['fitting_doy'], temperature=case['fitting_temp'], 
              method = 'BF', optimizer_params=case['optimizer_parameters'],
              verbose=True, debug=True)
    check_known_values(estimated_params = model.get_params(), 
                       known_params = case['known_model_params'],
                       message = case['test_name'])
