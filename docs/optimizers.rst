#################
Optimizer Methods
#################

To estimate parameters in models pyPhenology uses optimizers built-in to `scipy <https://docs.scipy.org/doc/scipy/reference/optimize.html#global-optimization>`__.
The most popular optimization technique in phenology is simulated annealing. This was implimented in scipy previously but was `dropped <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.anneal.html>`__
in favor of the basin hopping algorithm. Optimizers available are:

* Differential evolution (the default)
* Basin hopping
* Brute force


