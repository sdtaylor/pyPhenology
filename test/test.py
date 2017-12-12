from pyPhenology import utils, models

doy, temp = utils.load_test_data()

# Test with no fixed parameters
model = models.Thermal_Time()
model.fit(doy, temp)
model.predict(doy, temp)
# Test with 1 fixed parameters
#model = models.Thermal_Time()
#model.fit(doy, temp)
