from pyPhenology import utils
observations, temp = utils.load_test_data(name='vaccinium')
import pandas as pd
import time

models_to_use = ['ThermalTime','Alternating','Uniforc','MSB','Unichill']
methods_to_use = ['DE','BF']
sensible_defaults = ['testing','practical','intensive']

timings=[]
for m in models_to_use:
    for optimizer_method in methods_to_use:
        for s_default in sensible_defaults:
            model = utils.load_model(m)()
            
            start_time = time.time()
            model.fit(observations, temp, method=optimizer_method, optimizer_params=s_default)
            fitting_time = round(time.time() - start_time,2)
            
            num_parameters = len(model.get_params())
            
            timings.append({'model':m,
                            'num_parameters':num_parameters,
                            'method':optimizer_method,
                            'sensible_default':s_default,
                            'time':fitting_time})
            
            print(timings[-1])
            print('#######################')
    
timings = pd.DataFrame(timings)
timings.to_csv('model_fit_timings.csv')