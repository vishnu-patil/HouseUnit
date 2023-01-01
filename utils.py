import pickle
import pandas as pd
import numpy as np
import sklearn
import warnings
warnings.filterwarnings('ignore')

class House():
    def __init__(self,user_data):
        self.user_data = user_data

    def load_saved_data(self):
        with open(r"knnmodel.pkl",'rb') as f:
            self.model = pickle.load(f)

        with open(r"sacler.pkl",'rb') as r:
            self.normal = pickle.load(r)    

    def predict(self):

        self.load_saved_data()
        transaction_date = eval(self.user_data['transaction_date'])
        house_age = eval(self.user_data['house_age'])
        distance_to_the_nearest_MRT_station = np.log(eval(self.user_data['distance_to_the_nearest_MRT_station']))
        number_of_convenience_stores = eval(self.user_data['number_of_convenience_stores'])
        latitude = eval(self.user_data['latitude'])
        longitude = eval(self.user_data['longitude'])  

        test_array = np.zeros(len(self.model.feature_names_in_))
        test_array[0] = transaction_date
        test_array[1] = house_age
        test_array[2] = distance_to_the_nearest_MRT_station
        test_array[3] = number_of_convenience_stores
        test_array[4] = latitude
        test_array[5] = longitude

        inputs = self.normal.transform([test_array])
        print(inputs)         

        prediction = np.around(self.model.predict(inputs)[0],2)
        print("The Prediction Is:",prediction)
        return prediction

if __name__ == "__main__":
    obj = House
    obj        