#################
Optimizer Methods
#################

Parameters of phenology models have a complex search space and are commonly fit with `global optimization <https://en.wikipedia.org/wiki/Global_optimization>`__ algorithms.
To estimate parameters pyPhenology uses optimizers built-in to `scipy <https://docs.scipy.org/doc/scipy/reference/optimize.html#global-optimization>`__.
Optimizers available are:

* Differential evolution (the default)
* Basin hopping
* Brute force

Changing the optimizer
======================

The optimizer can be specified in the ``fit`` method by using the codes ``DE``, ``BH``, or ``BF`` for differential evolution, basin hopping, or brute force, respectively::

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

* ``testing``  Designed to be quick for testing code. Results from this should not be used for analysis.
* ``practical`` Default. Should produce realistic results on desktop systems in a relatively short period.
* ``intensive`` Designed to find the absolute optimal solution. Can potentially take hours to days on large datasets.


The 2nd is using a dictionary for customized optimizer arguments::

    model.fit(observations, temp, method='DE',optimizer_params={'popsize': 50,
                                                                'maxiter': 5000})


All the arguments in the scipy optimizer functions are available via the ``optimizer_params`` argument in ``fit``. The important
ones are described below, but also look at the available options in the scipy documentation. Any arguments not set will be
set to the default specifed in the scipy package. Note that the three presets can be used with all optimizer methods, but using
custimized methods will only work for that specific method. For example the ``popsize`` argument above will only work with method ``DE``.


Optimizers
==========

.. _optimizer_de:

Differential Evolution
----------------------
Differential evolution uses a population of models each randomly initialized to different parameter values within the respective search spaces.
Each "member" is adjusted slightly based on the performance of the best model. This process is repeated until the ``maxiter`` value is reached
or a convergence threshold is met.

`Differential evolution Scipy documentation <https://docs.scipy.org/doc/scipy-1.0.0/reference/generated/scipy.optimize.differential_evolution.html>`__

Key arguments
^^^^^^^^^^^^^
See the official documentation for more in depth details.

* ``maxiter`` : int
    the maximum number of itations. higher means potentially longer fitting times.
* ``popsize`` : int
    total population size of randomly initialized models. higher means longer fitting times.
* ``mutation`` : float, or tuple
    mutation constant. must be `0 < x < 2`. Can be a constant (float) or a range (tuple). Higher mean longer fitting times.
* ``recombination`` : float
    probability of a member progressing to the next generation. must be `0 < x < 1`. Lower means longer fitting times.
* ``disp`` : boolean
    Display output as the model is fit.

Presets
^^^^^^^
* ``testing``::

    {'maxiter':5,
     'popsize':10,
     'mutation':(0.5,1),
     'recombination':0.25,
     'disp':False}

* ``practical``::

    {'maxiter':1000,
     'popsize':50,
     'mutation':(0.5,1),
     'recombination':0.25,
     'disp':False}

* ``intensive``::

    {'maxiter':10000,
     'popsize':100,
     'mutation':(0.1,1),
     'recombination':0.25,
     'disp':False}

.. _optimizer_bh:

Basin Hopping
-------------
Basin hopping first makes a random guess at the parameters, then finds the local minima using `L-BFGS-B <https://docs.scipy.org/doc/scipy-1.0.0/reference/optimize.minimize-lbfgsb.html>`__.
From the local minima the parameters are then randomly perturbed and minimized again, with this new estimate accepted using a Metropolis criterion. This is repeated
until ``niter`` is reached. The parameters with the best score are selected.
Basin hopping is essentially the same as simulated annealing, but with the added step of finding the local minima between random perturbations.
Simulated annealing was retired from the Scipy packge in favor of basin hopping, see the note `here <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.anneal.html>`__.

`Basin Hopping Scipy documentation <https://docs.scipy.org/doc/scipy-1.0.0/reference/generated/scipy.optimize.basinhopping.html>`__

Key arguments
^^^^^^^^^^^^^
See the official documentation for more in depth details.

* ``niter`` : int
    the number of itations. higher means potentially longer fitting times.
* ``T`` : float
    The “temperature” parameter for the accept or reject criterion.
* ``stepsize`` : float
    The size of the random perturbations
* ``disp`` : boolean
    Display output as the model is fit.

Presets
^^^^^^^
* ``testing``::

    {'niter': 100,
     'T': 0.5,
     'stepsize': 0.5,
     'disp': False}

* ``practical``::

    {'niter': 50000,
     'T': 0.5,
     'stepsize': 0.5,
     'disp': False}

* ``intensive``::

    {'niter': 500000,
     'T': 0.5,
     'stepsize': 0.5,
     'disp': False}

.. _optimizer_bf:

Brute Force
-----------

Brute force is a comprehensive search within predefined parameter ranges.

`Brute force Scipy documentation <https://docs.scipy.org/doc/scipy-1.0.0/reference/generated/scipy.optimize.brute.html>`__

Key Arguments
^^^^^^^^^^^^^
See the official documentation for more details

* ``Ns`` : int
    Number of grid points within search space to search over. See below.
* ``finish`` : function
    Function to find the local best solution from the best search space solution. This is set to ``optimize.fmin_bfgs`` in the
    presets, which is the scipy bfgs minimizer. See more options
    `here <https://docs.scipy.org/doc/scipy-1.0.0/reference/optimize.html#local-optimization>`__.
* ``disp`` : boolean
    Display output as the model is fit.

Presets
^^^^^^^^
* ``testing``::

    {'Ns':2,
     'finish':optimize.fmin_bfgs,
     'disp':False}

* ``practical``::

    {'Ns':20,
     'finish':optimize.fmin_bfgs,
     'disp':False}

* ``intensive``::

    {'Ns':40,
     'finish':optimize.fmin_bfgs,
     'disp':False}

Brute Force Notes
^^^^^^^^^^^^^^^^^
The ``Ns`` argument defines the number of points to test with each search parameter. For example consider the following search spaces for a
three parameter model::

    {'t1': (-10,10), 'T':(0,10), 'F': (0,1000),}

Using ``Ns=20`` will search all combinations of::

    t1=[-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    T=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5]
    F=[0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950]

which results in :math:`20^3` model evalutaions. In this way model fitting time increases exponentially with the number of parameters in a model.

Alternatively you can set the search range using slices of (low, high, step) instead of (low,high). This allows for more control over the search space for
each paramter. For example::

     {'t1': slice(-10, 10, 1),'T': slice(0,10, 1),'F': slice(0,1000, 5)}

Note that using slices this way only works for the brute force method. This can create more realistic search space for each parameter.
But in this example the number of evalutaions is still high, :math:`20*10*200=40000`.
It's recommended that Brute Force is only used for models with a low number of parameters, otherwise Differential Evolution is
quicker and more robust.
