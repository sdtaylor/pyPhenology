########
Examples
########

.. _example_model_selection_aic:

Model selection via AIC
=======================

This will fit the vaccinium budburst data to 3 different models,
and choose the best performing one via AIC

.. literalinclude:: ../examples/model_selection_aic.py

Output:
::

    model ThermalTime got an aic of 55.51000634199631
    model Alternating got an aic of 60.45760650906022
    model Linear got an aic of 64.01178179718035
    Best model: ThermalTime
    Best model paramters:
    {'t1': 90.018369435585129, 'T': 2.7067432485899765, 'F': 181.66471096956883}

.. _example_model_rmse:

RMSE Evaluation
===============

This will calculate the root mean square error (RMSE) on 2 species,
each with a budburst and flower phenophase, using 2 models. Both are
Thermal Time models with a start date of 1 (Jan. 1), and the temperature
threshold is 0 for one and 5 for the other.

.. literalinclude:: ../examples/model_rmse.py

.. image:: https://i.imgur.com/vTdKOdO.png
