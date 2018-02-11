#################
Optimizer Methods
#################

To estimate parameters in models pyPhenology uses optimizers built-in to `scipy <https://docs.scipy.org/doc/scipy/reference/optimize.html#global-optimization>`__.
The most popular optimization technique in phenology research is simulated annealing. This was implimented in scipy previously but was `dropped <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.anneal.html>`__
in favor of the basin hopping algorithm. Optimizers available here are:

* Differential evolution (the default)
* Basin hopping
* Brute force

Adjust the optimizer
--------------------

The optimize can be in the ``fit`` method by using the codes ``DE``, ``BH``, or ``BF`` for differential evolution,
basin hopping, or brute force, respectively::

    from pyPhenology import models, utils
    observations, temp = utils.load_test_data(name='vaccinium')
    
    model = models.ThermalTime(parameters={})
    model.fit(observations, temp, method='BH')


All the arguments in the scipy optimizer functions are available via the ``optimizer_params`` argument in ``fit``.
Here we set arguments for differential evolution to have a population size of 50, and the max number of iterations to 5000::

    from pyPhenology import models, utils
    observations, temp = utils.load_test_data(name='vaccinium')
    
    model = models.ThermalTime(parameters={'t1':1})
    model.fit(observations, temp, method='DE',optimizer_params={'popsize': 50,
                                                                'maxiter': 5000})

Note that the optimizer parameters are not interchangable between the 3 methods. 

Sensible defaults have been setup for for the differential evolution and basin hopping for various scenarios. The 
details of which are listed below

* ``testing`` Optimizer parameters set to finish quickly. Designed for testing code.
* ``practical`` Should produce realistic results on desktop systems in a relatively short period. (name of this open to suggestions)
* ``exhaustive`` Designed to find the absolute optimal solution. Can potentially take hours to days.

