# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 02:57:42 2017

@author: Kenneth
"""

import graphlab
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

data = graphlab.SFrame('C:/Users/Kenneth/Downloads/train.csv')
data= data.fillna('LotFrontage', np.median(data['LotFrontage']))
data = data.fillna('GarageYrBlt', np.median(data['GarageYrBlt']))
data = data.fillna('MasVnrArea', np.median(data['MasVnrArea']))

features_list = data.column_names()
features_list.remove('Id')
features_list.remove('SalePrice')

data.show(view="Scatter Plot", x="sqft_living", y="price")
train_data,test_data = data.random_split(.8,seed=0)

sqft_model = graphlab.linear_regression.create(train_data, target='SalePrice', 
                                               features=['LotArea'],validation_set=None)
plt.plot(test_data['LotArea'],test_data['SalePrice'],'.',
        test_data['LotArea'],sqft_model.predict(test_data),'-')                                               

features = ['TotalBsmtSF', 'Neighborhood','BldgType', 'YrSold', 'Street'
, 'HouseStyle','YearBuilt', 'SaleCondition', 'SaleType']

features_model = graphlab.linear_regression.create(train_data,target='SalePrice',
                                                   features=features,validation_set=None)

print sqft_model.evaluate(test_data)
print features_model.evaluate(test_data)


boost_sqft = graphlab.boosted_trees_regression.create(train_data, target='SalePrice', 
                                               features=['LotArea'],validation_set=None)
                                               
boost_features = graphlab.boosted_trees_regression.create(train_data,target='SalePrice',
                                                   features=features,validation_set=None)

#Prediction for House 1
house1 = data[data['Id'] == 1]
print features_model.predict(house1)

f_predictions = features_model.predict(data)
b_predictions = boost_features.predict(data)

from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(data['SalePrice'], b_predictions))
