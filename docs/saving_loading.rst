.. _saving_loading:

=========================
Saving and loading models
=========================

Parameters from a model can be obtained in a dictionary via the :any:`Model.get_params` method::
    
    model.get_params()

Fitted models can also be saved to a file::

    model.save_params(filename='model_1_parameters.json')
    
Paremeters are saved to a json file, though the json extension isn't required.   

Saved model files can be loaded again by passing the saved filename as the ``parameters`` argument 
in the model initialization::

    model = models.ThermalTime(parameters = 'model_1_parameters.json')

Note that this requires the loaded model to match the saved model. You
can also use the generic loader :any:`utils.load_saved_model` ::

    model = utils.load_saved_model('model_1_parameters.json')
    model.get_params()
    {'t1': 4.9538373877994291, 'F': 270.006971948699, 'T': 5}
