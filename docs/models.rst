.. currentmodule:: pyPhenology.models

======
Models
======

.. _model_api:

Model API
---------
All models use the same methods for fitting, prediction, and saving.

.. autosummary::
    :toctree: generated/
    
    Model.fit
    Model.predict
    Model.score
    Model.save_params
    Model.get_params

Primary Models
--------------
.. autosummary::
    :toctree: generated/
    
    ThermalTime
    Alternating
    Uniforc
    Unichill
    Linear
    MSB
    Sequential
    M1
    Naive



Ensemble Models
---------------
.. autosummary::
    :toctree: generated/
    
    BootstrapModel
    WeightedEnsemble
