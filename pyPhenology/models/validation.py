import pandas as pd

def validate_temperature(temperature):
    """ Validate a temperature dataframe to the format used in this package.
    
    Parameters
    ----------
    temperature : Pandas Dataframe
    
    Returns
    -------
    temperature : The same dataframe but with only the valid columns
    """
    assert isinstance(temperature, pd.DataFrame), 'temperature should be a pandas dataframe'
    valid_columns = ['temperature','year','site_id']
    for column in valid_columns:
        assert column in temperature.columns, 'missing required temperature column: '+column
    
    return temperature[valid_columns]

def validate_DOY(DOY, for_prediction=False):
    """ Validate a DOY dataframe to the format used in this package.
    
    Parameters
    ----------
    DOY : Pandas Dataframe
    
    for_prediction : bool
        If being used to in model.predict(), then one less colum is required
        
    Returns
    -------
    DOY : The same dataframe but with only the valid columns
    """
    assert isinstance(DOY, pd.DataFrame), 'DOY should be a pandas dataframe'
    valid_columns = ['year','site_id']
    if not for_prediction: valid_columns.append('doy')
    
    for column in valid_columns:
        assert column in DOY.columns, 'missing required DOY column: '+column
    
    return DOY[valid_columns]


def validate_model(model_class):
    required_attributes = ['_apply_model','all_required_parameters']
    for attribute in required_attributes:
        assert hasattr(model_class, attribute), 'Missing model attribute: ' + str(attribute)