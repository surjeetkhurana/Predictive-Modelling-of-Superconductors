 # -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 15:38:28 2023

@author: Surjeet
"""




import pandas as pd
import numpy as np

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from statistics import mean

from tensorflow.keras import backend as K
#from tensorflow.keras import backend as K
def root_mean_squared_error(y_true, y_pred): 
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

data = pd.read_csv("Ehull_TrainSet.csv")
data = data.dropna()

data = data.reset_index(drop=True)

targets = data["EboveHull"]


X = data.drop("Filename", axis = 1)
X = X.drop("MP_ID", axis = 1)
X = X.drop("nElements", axis = 1) 
X = X.drop("Material", axis = 1)
X = X.drop("Count", axis = 1)
X = X.drop("EboveHull", axis = 1)
X = X.drop("FormationEnergy", axis = 1)


num_folds = 10
inputs=pd.DataFrame(X)

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True , random_state = 1)  


df = []
df1 = []
# K-fold Cross Validation model evaluation
"""
#use this part of the code when training the initial model, keep it within the commented section if just want to predict
model =  ExtraTreesRegressor(n_estimators =  1000, min_samples_split= 2, min_samples_leaf= 4, max_depth= 30,
     max_features=1.0, bootstrap= False)
    
model.fit(inputs, targets)


import pickle
filename = 'ExtraTrees86%Accuracy.sav'
pickle.dump(model, open(filename, 'wb'))

    
#y_pred = model.predict(inputs)
"""


import pickle
model = pickle.load(open("Extra_Trees.sav", 'rb'))

data1 = pd.read_csv("Ehull_TestSet.csv")
data1 = data1.dropna()

data1 = data1.reset_index(drop=True)





targets1 = data1["EboveHull"]


X1 = data1.drop("Filename", axis = 1)
X1= X1.drop("MP_ID", axis = 1)
X1= X1.drop("nElements", axis = 1) 
X1 = X1.drop("Material", axis = 1)
X1 = X1.drop("Count", axis = 1)
X1 = X1.drop("EboveHull", axis = 1)
X1 = X1.drop("FormationEnergy", axis = 1)



y_pred = model.predict(X1)
  
from sklearn.metrics import r2_score
rs = r2_score
Rsquare = rs(targets1, y_pred)
    
from sklearn.metrics import mean_absolute_error
rs = mean_absolute_error
mae = rs(targets1, y_pred)

from sklearn.metrics import mean_squared_error
rs = mean_squared_error
mse = rs(targets1, y_pred)
         
    
    
one = np.ravel(targets1)
two = np.ravel(y_pred)
df.insert(1,one)
df1.insert(1,two)
    
   

"""
dff = pd.DataFrame(df) 
writer = pd.ExcelWriter('ETTestFinal.xlsx', engine='xlsxwriter')
dff.to_excel(writer, sheet_name='RandomForest', index=False)
writer.save()

dff1 = pd.DataFrame(df1)
writer = pd.ExcelWriter('ETPredFinal.xlsx', engine='xlsxwriter')
dff1.to_excel(writer, sheet_name='RandomForest', index=False)
writer.save()
"""
