from pyPhenology import utils, models

doy, temp = utils.load_test_data()

model_list={'Uniforc': models.Uniforc,
            'Unichill': models.Unichill,
            'Thermal_Time':models.Thermal_Time}

for model_name, Model in model_list.items():
    #Test with no fixed parameters
    print(model_name + ' - Estimate all parameters')
    model=Model()
    model.fit(DOY=doy, temperature=temp, verbose=True)
    model.predict(doy, temp)

    # Test with 1 fixed parameters
    #model = models.Thermal_Time()
    #model.fit(doy, temp)