#########
Utilities
#########

Model Loading
=============
Models can be loaded individually using the base classes::

    from pyPhenology import models
    model1 = models.ThermalTime()
    model2 = models.Alternating()

or with a string of the same name via ``load_model``. Note that
they must be initialized after loading, which allows you to :ref:`set the parameters <setting_parameters>`::

    from pyPhenology import utils
    Model1 = utils.load_model('ThermalTime')
    Model2 = utils.load_model('Alternating')
    model1 = Model1()
    model2 = Model2()

The :any:`BootstrapModel` must still be loaded directly. But it can accept core models loaded via ``load_model``::

    model = models.BootstrapModel(core_model = utils.load_model('Alternating'))

Test Data
=========
Two sets of observations are available for use in the package as well as associated 
mean daily temperature derived from the PRISM dataset. The data
is in pandas ``data.frames`` as outlined in :ref:`data structures <data_structure>`.

The first is observations of `Vaccinium corymbosum` from Harvard Forest, with both
flower and budburst phenophases.
is observations of `Populus tremuloides` (Aspen) from the National Phenology Network.
Both have flower and budburst phenophases available. 

::

    from pyPhenology import utils
    observations, predictors = utils.load_test_data(name='vaccinium',
                                                    phenophase='budburst')

    observations.head()

                    species  site_id  year  doy  phenophase
    0  vaccinium corymbosum        1  1991  100         371
    1  vaccinium corymbosum        1  1991  100         371
    2  vaccinium corymbosum        1  1991  104         371
    3  vaccinium corymbosum        1  1998  106         371
    4  vaccinium corymbosum        1  1998  106         371
    
    predictors.head()

       site_id  temperature    year  doy
    0        1        -3.86  1989.0  0.0
    1        1        -4.71  1989.0  1.0
    2        1        -1.56  1989.0  2.0
    3        1        -7.88  1989.0  3.0
    4        1       -15.24  1989.0  4.0

    observations, predictors = utils.load_test_data(name='vaccinium',
                                                    phenophase='flowers')

                     species  site_id  year  doy  phenophase
    48  vaccinium corymbosum        1  1998  122         501
    49  vaccinium corymbosum        1  1998  122         501
    50  vaccinium corymbosum        1  1991  124         501
    51  vaccinium corymbosum        1  1991  124         501
    52  vaccinium corymbosum        1  1998  126         501


The second is `Populus tremuloides` (Aspen) from the National Phenology Network,
and also has flowers and budburst phenophases available. This has observations
across many sites, making it a suitable test for spatially explicit models.

::
    
    observations, predictors = utils.load_test_data(name='aspen',
                                                    phenophase='budburst')
                                                    
    observations.head()
                   species  site_id  year  doy  phenophase
    0  populus tremuloides    16374  2014   44         371
    1  populus tremuloides     2330  2011   47         371
    2  populus tremuloides     2330  2010   48         371
    3  populus tremuloides     1020  2009   73         371
    4  populus tremuloides    22332  2016   72         371
