import json
import pickle
import numpy as np
import warnings
warnings.simplefilter(action='ignore')

__locations = None
__data_columns = None
__model = None


def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1
    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bhk
    x[2] = bath
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0], 2)

def get_location_names():
    load_saved_artifacts()
    return __locations

# using global keyword so that the variables are treated as global
# variables
def load_saved_artifacts():
    print("loading artifacts started")
    global __data_columns
    global __locations
    with open('artifacts/columns.json', 'rb') as f:
# data_columns is the key used in the json file.
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    global __model
    with open("artifacts/Real_Estate_prices_model.pickle", "rb") as f:
        __model = pickle.load(f)
    print("Loading the artifacts is done.")

if __name__ == "__main__":
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
    print(get_estimated_price('Kalhalli', 1000, 2, 2)) # other location