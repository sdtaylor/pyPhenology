from pyPhenology import utils
import numpy as np

models_to_test = ['ThermalTime','Alternating','Linear']

observations, temp = utils.load_test_data(name='vaccinium')

# Only keep the leaf phenophase
observations = observations[observations.phenophase==371]

observations_test = observations[0:10]
observations_train = observations[10:]

# AIC based off mean sum of squares
def aic(obs, pred, n_param):
    return len(obs) * np.log(np.mean((obs - pred)**2)) + 2*(n_param + 1)

best_aic=np.inf
best_base_model = None
best_base_model_name = None

for model_name in models_to_test:
    Model = utils.load_model(model_name)
    model = Model()
    model.fit(observations_train, temp, optimizer_params='practical')
    
    model_aic = aic(obs = observations_test.doy.values,
                    pred = model.predict(observations_test,
                                         temp),
                    n_param = len(model.get_params()))
    
    if model_aic < best_aic:
        best_model = model
        best_model_name = model_name
        best_aic = model_aic
        
    print('model {m} got an aic of {a}'.format(m=model_name,a=model_aic))
    
print('Best model: {m}'.format(m=best_model_name))
print('Best model paramters:')
print(best_model.get_params())