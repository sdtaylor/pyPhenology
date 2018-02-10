================================
Controlling Parameter Estimation
================================

By default all parameters in the models are free to vary within their predefined search ranges. 
The default search ranges are predefined based on being applicable to a wide variety of plants.
Initial parameters can be adjusted in two ways.

* The search range can be ajdusted.
* Any or all parameters can be set to a fixed value

Both of these are done via the parameters argument in the initial model call.

Setting parameters to fixed values
----------------------------------

Here is a common example, where in the thermal time model ``t1``, the day when warming accumulation begins,
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

One more example where the Uniforc model is set to a ``t1`` of 60 (about March 1), and the other parameters are estimated::

    model = models.Uniforc(parameters={'t1':1})
    model.fit(observations, temp)
    model.get_params()
    
    {'F': 11.050063297905695, 'b': -2.0395193186815908, 'c': 9.3016675933620956, 't1': 1}


Setting a search range for parameters
-------------------------------------

To specify a different search range for a parameter use tuples with a high and low value. These can be
mixed and matched with setting fixed values.

For example the Thermal Time model with narrow search range for ``t1`` and ``F`` but ``T`` fixed at 5 degrees C::

    model = models.ThermalTime(parameters={'t1':(-10,10), 'F':(100,500),'T':5})
    model.fit(observations, temp)
    model.get_params()
    
    {'t1': 4.9538373877994291, 'F': 270.006971948699, 'T': 5}
    
The above works for the optimization methods Differential Evolution (the default), Basin Hopping, and Simulated Annealing.  
For the brute force method you must specify slice.

TODO
