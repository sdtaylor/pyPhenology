import pandas as pd
import pkg_resources

def load_test_data(name='vaccinium'):
    """Pre-loaded phenology and associated
    temperature data.

    Available datasets
    ------------------
    'vaccinium'
        Vaccinium corymbosum phenology from Harvard Forest
        Both flowers (phenophase 501) and leaves (phenophase 371)
    
    Parameters
    ----------
    name : str
        Name of the test dataset
    
    Returns
    -------
    doy, temp : tuple
        Pandas dataframes of phenology observations
        and associated temperatures.
    """
    if name=='vaccinium':
        doy_file = 'data/vaccinium_doy.csv'
        temp_file= 'data/vaccinium_temperature.csv'
    else:
        raise ValueError('Uknown dataset name: ' + str(name))
        
    doy_file = pkg_resources.resource_stream(__name__, doy_file)
    temp_file = pkg_resources.resource_stream(__name__, temp_file)
    doy = pd.read_csv(doy_file)
    temp= pd.read_csv(temp_file)
    return doy, temp
