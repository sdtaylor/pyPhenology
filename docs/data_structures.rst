.. _data_structure:

##############
Data Structure
##############

Your data must be structured in a specific way to be used in the package.

Phenology Observation Data
^^^^^^^^^^^^^^^^^^^^^^^^^^
Observation data consists of the following

* doy: These are the julian date (1-365) of when a specific phenological event happened. 
* site_id: A site identifier for each doy observation
* year: A year identifier for each doy observation

These should be structured in columns in a pandas data.frame, where every row is a 
single observation. For example the built in vaccinium dataset looks like this::

    from pyPhenology import models, utils
    observations, temp = utils.load_test_data(name='vaccinium')
    
    obserations.head()
    
                    species  site_id  year  doy  phenophase
    0  vaccinium corymbosum        1  1991  100         371
    1  vaccinium corymbosum        1  1991  100         371
    2  vaccinium corymbosum        1  1991  104         371
    3  vaccinium corymbosum        1  1998  106         371
    4  vaccinium corymbosum        1  1998  106         371

There are extra columns here for the species and phenophase, those will be ignored inside
the pyPhenology package. 


Phenology Environmental Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The majority of the models use only daily mean temperature as a driver. 
This is required for for every day of the winter and spring leading up to the phenophase event.
The predictors ``data.frame`` should have the following structure.

* site_id: A site identifier for each location. 
* year: The year of the temperature timeseries
* temperature: The observed daily mean temperature in degrees Celcius.
* doy: The julian date of the mean temperature

These should columns be in a ``data.frame`` like the observations. The example vaccinium
dataset has temperature observations::

    predictors.head()
        site_id  temperature  year  doy  latitude  longitude  daylength
    0        1        -3.86  1989    0   42.5429   -72.2011       8.94
    1        1        -4.71  1989    1   42.5429   -72.2011       8.95
    2        1        -1.56  1989    2   42.5429   -72.2011       8.97
    3        1        -7.88  1989    3   42.5429   -72.2011       8.98
    4        1       -15.24  1989    4   42.5429   -72.2011       9.00

Note than any other columns in the predictors ``data.frame`` besides the ones
used will be ignored.

Currently two other models use other predictors besides daily mean temerature.
The :any:`M1` uses daylength as a predictor as well as daily mean temperature. 
The predictors ``data.frame`` should thus have a daylength column in addition 
to the temperature as shown above. 

The :any:`Naive` model uses only latitude in it's calculation and thus requires
a predictors ``data.frame`` with the latitude for every site. For example::

    predictors.head()

       site_id   latitude
    0      258  39.184269
    1      414  44.277962
    2      475  47.027077
    3      637  44.340950
    4      681  41.296783

On the Julian Date
^^^^^^^^^^^^^^^^^^^
The julian date (usually referenced as DOY for "day of year") is used throughout 
the package. This can be negative if referencing something from the prior season. 
For example consider the following data from the aspen dataset::

    predictors.head()
    
       site_id  temperature  year  doy   latitude   longitude  daylength
    0      258         6.28  2009  -67  39.184269 -106.854614      10.52
    1      414         8.12  2009  -67  44.277962  -70.315315      10.22
    2      475         5.30  2009  -67  47.027077 -114.049248      10.04
    3      637         8.30  2009  -67  44.340950  -72.461220      10.22
    4      681         9.85  2009  -67  41.296783 -105.574600      10.40

The ``doy`` -67 here refers to Oct. 26 for the growing year 2009. Formating dates in
this fashion allows for a continuous range of numbers across years, and is common
in phenology studies. 

January 1 will always be DOY 1. 

Notes
^^^^^
* If you have only a single site, make a "dummy" site_id column set to 1 for both temperature and
  observation dataframes.
* If you have only a single year then it still must be represented in the year column of both data.frames.
