import pandas as pd
import pkg_resources

def load_test_data():
    doy_file = pkg_resources.resource_stream(__name__, 'data/vaccinium_doy.csv')
    temp_file = pkg_resources.resource_stream(__name__, 'data/vaccinium_temperature.csv')
    doy = pd.read_csv(doy_file)
    temp= pd.read_csv(temp_file)
    return doy, temp