.. _controlling_parameter_estimation:

================================
Controlling Parameter Estimation
================================

By default all parameters in the models are free to vary within their predefined search ranges. 
The default search ranges are predefined based on being applicable to a wide variety of plants.
Initial parameters can be adjusted in two ways.

* The search range can be ajdusted.
* Any or all parameters can be set to a fixed value

Both of these are done via the ``parameters`` argument in the initial model call.

.. _setting_parameters:

Setting parameters to fixed values
----------------------------------

Here is a common example, where in the :any:`ThermalTime` model ``t1``, the day when warming accumulation begins,
is set to Jan. 1 (doy 1) by setting it to an integer. The other two parameters, ``F`` and ``T``, and then estimated::

    from pyPhenology import models, utils
    observations, temp = utils.load_test_data(name='vaccinium')
    
    model = models.ThermalTime(parameters={'t1':1})
    model.fit(observations, temp)
    model.get_params()
    
    {'T': 8.6286577557177608, 'F': 156.76212563809247, 't1': 1}


Similarly, we can also set the temperature threshold ``T`` to fixed values. Then only ``F``, the total degree days required, 
is estimated::

    model = models.ThermalTime(parameters={'t1':1,'T':5})
    model.fit(observations, temp)
    model.get_params()
    
    {'F': 274.29110894742541, 't1': 1, 'T': 5}
    
Note that if you set all the parameters of a model to fixed values then no fitting can be done::

    model = models.ThermalTime(parameters={'t1':1,'T':5, 'F':50})
    model.fit(observations, temp)
    
    RuntimeError: No parameters to estimate

One more example where the :any:`Uniforc` model is set to a ``t1`` of 60 (about March 1), and the other parameters are estimated::

    model = models.Uniforc(parameters={'t1':60})
    model.fit(observations, temp)
    model.get_params()
    
    {'F': 11.259894714800524, 'b': -3.1259822030672773, 'c': 9.1700700063424012, 't1': 60}


Setting a search range for parameters
-------------------------------------

To specify a different search range for a parameter use tuples with a low and high value. These can be
mixed and matched with setting fixed values.

For example the Thermal Time model with narrow search range for ``t1`` and ``F`` but ``T`` fixed at 5 degrees C::

    model = models.ThermalTime(parameters={'t1':(-10,10), 'F':(100,500),'T':5})
    model.fit(observations, temp)
    model.get_params()
    
    {'t1': 4.9538373877994291, 'F': 270.006971948699, 'T': 5}
    
The above works well for the optimization method :ref:`optimizer_de` (the default).
For the brute force method you can also specify a slice in the form (low, high, step), see :ref:`optimizer_bf`

.. _parameter_saving_loading:

Saving and loading model parameters
-----------------------------------

Fitted parameters from a model can be obtained in a dictionary via the :any:`Model.get_params` method as shown above.
They can also be saved to a file::

    model.save_params(filename='model_1_parameters.csv')
    
Paremeters are saved to a csv file, though the csv extension isn't required.   

Saved parameter files can be loaded again by passing the saved filename as the ``parameters`` argument 
in the model initialization::

    model = models.ThermalTime(parameters = 'model_1_parameters.csv')
    
