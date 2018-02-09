##############
Data Structure
##############

Your data must be structured in a specific way to be used in the package.

Phenology Observation Data
^^^^^^^^^^^^^^^^^^^^^^^^^^
Observation data consists of the following

* doy: These are the julien date (1-365) of when a specific phenological event happened. 
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
The current models only support daily mean temperature as a driver. Models require the daily
temperature for every day of the winter and spring leading up to the phenophase event

* site_id: A site identifier for each location. 
* year: The year of the temperature timeseries
* temperatuer: The observed daily mean temperature in degrees Celcius.
* doy: The julien date of the mean temperature

These should columns in a data.frame like the observations. The example vaccinium
dataset has temperature observations::

       site_id  temperature    year  doy
    0        1        -3.86  1989.0  0.0
    1        1        -4.71  1989.0  1.0
    2        1        -1.56  1989.0  2.0
    3        1        -7.88  1989.0  3.0
    4        1       -15.24  1989.0  4.0


On the Julien Date
^^^^^^^^^^^^^^^^^^^
TODO
Jan. 1 is 0, but prior dates of the same winter are negative numbers. 

Notes
^^^^^
* If you have only a single site, make a "dummy" site_id column set to 1 for both temperature and
  observation dataframes.
* If you have only a single year
