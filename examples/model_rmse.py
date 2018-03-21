from pyPhenology import utils, models
import numpy as np
import pandas as pd

datasets_to_use = ['vaccinium','aspen']
phenophases_to_use = ['budburst','flowers']

num_boostraps=5

# Two Thermal Time models with fixed start day of Jan 1, and 
# with different fixed temperature thresholds.
# Each getting variation using 50 bootstraps.
bootstrapped_tt_model_1 = models.BootstrapModel(core_model=models.ThermalTime,
                                                num_bootstraps=num_boostraps,
                                                parameters={'t1':1,
                                                            'T':0})

bootstrapped_tt_model_2 = models.BootstrapModel(core_model=models.ThermalTime,
                                                num_bootstraps=num_boostraps,
                                                parameters={'t1':1,
                                                            'T':5})

models_to_fit = {'TT Temp 0':bootstrapped_tt_model_1,
                 'TT Temp 5':bootstrapped_tt_model_2}

results = pd.DataFrame()

for dataset in datasets_to_use:
    for phenophase in phenophases_to_use:
        
        observations, predictors = utils.load_test_data(name=dataset,
                                                        phenophase=phenophase)
        
        # Setup 20% train test split using pandas methods
        observations_test = observations.sample(frac=0.2,
                                                random_state=1)
        observations_train = observations[~observations.index.isin(observations_test.index)]
        
        observed_doy = observations_test.doy.values
        
        for model_name, model in models_to_fit.items():
            model.fit(observations_train, predictors, optimizer_params='practical')
            
            # Using aggregation='none' in BoostrapModel predict
            # returns results for all bootstrapped models in an
            # (num_bootstraps, n_samples) array. This will calculate
            # the RMSE of each model and var variation around that. 
            predicted_doy = model.predict(observations_test, predictors, aggregation='none')

            rmse = np.sqrt(np.mean( (predicted_doy - observed_doy)**2, axis=1))
            
            results_this_set = pd.DataFrame()
            results_this_set['rmse'] = rmse
            results_this_set['dataset'] = dataset
            results_this_set['phenophase'] = phenophase
            results_this_set['model'] = model_name

            results = results.append(results_this_set, ignore_index=True)


from plotnine import *

(ggplot(results, aes(x='model', y='rmse')) + 
     geom_boxplot() + 
     facet_grid('dataset~phenophase', scales='free_y'))