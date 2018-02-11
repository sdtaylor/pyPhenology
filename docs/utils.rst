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
A set of observations for `Vaccinium corymbosum` from the from Harvard Forest is available for use
in the package as well as associated mean daily temperature derived from the PRISM dataset. The data
is in pandas ``data.frames`` as outlined in :ref:`data structures <data_structure>`.

::

    from pyPhenology import utils
    observations, temp = utils.load_test_data(name='vaccinium')

    obserations.head()

                    species  site_id  year  doy  phenophase
    0  vaccinium corymbosum        1  1991  100         371
    1  vaccinium corymbosum        1  1991  100         371
    2  vaccinium corymbosum        1  1991  104         371
    3  vaccinium corymbosum        1  1998  106         371
    4  vaccinium corymbosum        1  1998  106         371
    
    temp.head()

       site_id  temperature    year  doy
    0        1        -3.86  1989.0  0.0
    1        1        -4.71  1989.0  1.0
    2        1        -1.56  1989.0  2.0
    3        1        -7.88  1989.0  3.0
    4        1       -15.24  1989.0  4.0
