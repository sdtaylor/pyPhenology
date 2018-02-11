#################
Optimizer Methods
#################

Parameters of phenology models have a complex search space and are commonly fit with `global optimization <https://en.wikipedia.org/wiki/Global_optimization>`__ algorithms. 
To estimate parameters pyPhenology uses optimizers built-in to `scipy <https://docs.scipy.org/doc/scipy/reference/optimize.html#global-optimization>`__.
Optimizers available here are:

* Differential evolution (the default)
* Brute force

Changing the optimizer
======================

The optimize can specified be in the ``fit`` method by using the codes ``DE``, or ``BF`` for differential evolution or brute force, respectively::

    from pyPhenology import models, utils
    observations, temp = utils.load_test_data(name='vaccinium')
    
    model = models.ThermalTime(parameters={})
    model.fit(observations, temp, method='DE')


Optimizer Arguments
===================

Arguments to the optimization algorithm are crucial to model fitting. These control things like the maximimum number of iterations and how to specify convergence.
Ideally one should choose arguments which find the "true" solution, yet this is a tradeoff of time and resources available. Models with a large number of parameters (such as the Unichill model) can take
several hours to days to fit if the optimizer arguments are set liberally. 

Optimizer arguments can be set two ways. The first is using some preset defaults::

    model.fit(observations, temp, method='DE', optimizer_params='practical')

* ``testing``  Designed for testing code. Results from this should not be used for analysis. 
* ``practical`` Default. Should produce realistic results on desktop systems in a relatively short period. (name of this open to suggestions)
* ``exhaustive`` Designed to find the absolute optimal solution. Can potentially take hours to days.


The 2nd is using a dictionary for customized optimizer arguments::

    model.fit(observations, temp, method='DE',optimizer_params={'popsize': 50,
                                                                'maxiter': 5000})
                                                                
                                                                
All the arguments in the scipy optimizer functions are available via the ``optimizer_params`` argument in ``fit``. The important
ones are described below, but also look at the available options in the scipy documentation. Any arguments not set will be
set to the default specifed in the scipy package. Note that the preset defaults can be used with all optimizer methods, but using
custimized methods will only work with a specific argument. For example the ``popsize`` argument above will only work with method ``DE``.


Optimizers
==========

Differential Evolution
----------------------
Differential evolution uses a population of models each randomly initialized to different parameter values within the respective search spaces. 
Each "member" is adjusted slightly based on the performance of the best model.

`Differential evolution Scipy documentation <https://docs.scipy.org/doc/scipy-1.0.0/reference/generated/scipy.optimize.differential_evolution.html>`__

Key arguments
^^^^^^^^^^^^^
See the official documentation for more details  

* ``maxiter`` : int
    the maximum number of itations. higher means potentially longer fitting times. 
* ``popsize`` : int
    total population size of randomly initialized models. higher means longer fitting times.
* ``mutation`` : float, or tuple
    mutation constant. must be `0 < x < 2`. Can be a constant (float) or a range (tuple). Higher mean longer fitting times.
* ``recombination`` : float
    probability of a member progressing to the next generation. must be `0 < x < 1`. Lower means longer fitting times.

Presets
^^^^^^^
* ``testing``
    
    * ``maxiter`` : 5
    * ``popsize`` : 10
    * ``mutation`` : (0.5,1)
    * ``recombination`` : 0.25

* ``practical``
    
    * ``maxiter`` : 1000
    * ``popsize`` : 50
    * ``mutation`` : (0.5,1)
    * ``recombination`` : 0.25

* ``intensive``
    
    * ``maxiter`` : 10000
    * ``popsize`` : 100
    * ``mutation`` : (0.1,1)
    * ``recombination`` 0.25



Brute Force
-----------

Brute force is a comprehensive search within predefined parameter ranges. 

`Brute force Scipy documentation <https://docs.scipy.org/doc/scipy-1.0.0/reference/generated/scipy.optimize.brute.html>`__

Key Arguments
^^^^^^^^^^^^^
See the official documentation for more details  

* ``Ns`` : int
    Number of grid points within search space to search over.
* ``finish`` : function
    Function to find the local best solution from the best search space solution. This is set to ``optimize.fmin_bfgs`` in the
    presets, which is the scipy bfgs minimizer. See more options 
    `here <https://docs.scipy.org/doc/scipy-1.0.0/reference/optimize.html#local-optimization>`__. 


Presets
^^^^^^^^
* ``testing``

    * ``Ns`` : 2
    * ``finish`` : ``optimize.fmin_bfgs``
    
* ``practical``

    * ``Ns`` : 20
    * ``finish`` : ``optimize.fmin_bfgs``

* ``intensive``

    * ``Ns`` : 40
    * ``finish`` : ``optimize.fmin_bfgs``


Brute Force Notes (could use some clearing up)
^^^^^^^^^^^^^^^^^
Using the brute force method with defaults for the Uniforc model like so::

    from pyPhenology import models, utils
    observations, temp = utils.load_test_data(name='vaccinium')

    model = models.Uniforc()
    model.fit(observations, temp, method='BF')

Will use the default optimizer arguments listed under ``practical`` with the default Uniforc search space of  ``{{'t1':(-67,298),'F':(0,200),'b':(-20,0),'c':(-50,50)}``.
The ``Ns`` argument of 20 will create 20 grid points equally spaced within the (low,high) search range of each parameter. Thus the total model evaluations are
equal to :math:`4 * 20 = 120`. A high ``Ns`` value along with a large number of parameters to estimate can create extremely long fitting times.


Alternatively you can set the search range using slices instead of (low,high)::

     {'t1': slice(-67,298,1),'F': slice(0,200,1),'b': slice(-20,0,1),'c': slice(-50,50,2)}
     
Slices are in the form (low, high, step). This creates are more realistic search space for each parameter. ``t1`` has units of days so 
the step should not be greater than 1. ``T`` and ``F`` are in degrees so searching beyond a step of 0.1 will likely not improve results
considerably. The total number of model evaluations for using this setup is :math:`(375)`






