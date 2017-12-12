import pandas as pd
import pkg_resources

def load_test_data(name='vaccinium'):
    if name=='vaccinium':
        doy_file = 'data/vaccinium_doy.csv'
        temp_file= 'data/vaccinium_temperature.csv'
    else:
        raise Exception('Uknown dataset name: ' + str(name))
        
    doy_file = pkg_resources.resource_stream(__name__, doy_file)
    temp_file = pkg_resources.resource_stream(__name__, temp_file)
    doy = pd.read_csv(doy_file)
    temp= pd.read_csv(temp_file)
    return doy, temp