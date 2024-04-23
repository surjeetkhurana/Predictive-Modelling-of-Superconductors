

import numpy as np
import tensorflow as tf
import keras
from sklearn.model_selection import KFold
import pandas as pd
# Load and preprocess your large dataset
# large_data, large_targets = load_large_tabular_dataset()

# Load and preprocess your smaller dataset
# small_data, small_targets = load_small_tabular_dataset()


keras.utils.set_random_seed(602)

large_data = pd.read_csv("MPdataWithBandgap.csv")
large_data = large_data.dropna()


large_targets = large_data["Bandgap"]


large_data= large_data.drop("Filename",axis = 1)
large_data= large_data.drop("Count",axis = 1)
large_data= large_data.drop("Bandgap",axis = 1)



# demonstrate data standardization with sklearn
from sklearn.preprocessing import StandardScaler
# load data
data = ...
# create scaler
scaler = StandardScaler()
# fit and transform in one step
large_data = scaler.fit_transform(large_data)
 

train_data = pd.read_excel("TrainSet.xlsx")
train_data = train_data.dropna()


train_targets = train_data["Average_CT"]

train_data= train_data.drop("Material",axis = 1)
train_data= train_data.drop("critical_temp",axis = 1)
train_data= train_data.drop("MP_id",axis = 1)
train_data= train_data.drop("Filename",axis = 1)
train_data= train_data.drop("Count",axis = 1)
train_data= train_data.drop("Average_CT",axis = 1)

column_names = list(train_data.columns.values)

test_data = pd.read_excel("TestSet.xlsx")
test_data = test_data.dropna()


test_targets = test_data["Average_CT"]

test_data= test_data.drop("Material",axis = 1)
test_data= test_data.drop("critical_temp",axis = 1)
test_data= test_data.drop("MP_id",axis = 1)
test_data= test_data.drop("Filename",axis = 1)
test_data= test_data.drop("Count",axis = 1)
test_data= test_data.drop("Average_CT",axis = 1)



# demonstrate data standardization with sklearn
#from sklearn.preprocessing import StandardScaler

"""
# create scaler
scaler = StandardScaler()
# fit and transform in one step
small_data = scaler.fit_transform(small_data)
"""

 
"""

# Define the architecture for the model
https://images.edrawsoft.com/articles/draw-neural-network/method2-2.png

# Compile the model for the initial training on the large dataset
model.compile(
    optimizer='adam', loss='mean_squared_error', metrics=['mae']
)

# Train the model on the large dataset
model.fit(
    large_data, large_targets,
    epochs=200, batch_size=32, validation_split=0.2
)

model.save("base_model.keras")
"""



# demonstrate data standardization with sklearn
from sklearn.preprocessing import StandardScaler
# create scaler
scaler = StandardScaler()
# fit and transform in one step
train_data = scaler.fit_transform(train_data)


scaler = StandardScaler()
# fit and transform in one step
test_data = scaler.fit_transform(test_data)

TL_model = keras.models.load_model("base_model.keras")

# Unfreeze layers for fine-tuning
for layer in TL_model.layers[:-1]:
    layer.trainable = True

my_layer = TL_model.get_layer('dense_647')
my_layer.trainable = True



opt = keras.optimizers.Adam(learning_rate=0.0001)

# Recompile the model for fine-tuning
TL_model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mae']
)

# Train the model on the smaller dataset
TL_model.fit(
    train_data, train_targets,
    epochs=500, batch_size=32, validation_data=(test_data, test_targets)
)

# Evaluate the model on the validation set

val_loss, val_mae = TL_model.evaluate(test_data, test_targets)


y_pred = TL_model.predict(test_data)

from sklearn.metrics import r2_score 
rs = r2_score
Rsquare = rs(test_targets,y_pred)


from sklearn.metrics import mean_absolute_error
rs = mean_absolute_error
mae = rs(test_targets,y_pred)

from sklearn.metrics import mean_squared_error
rs = mean_squared_error
mse = rs(test_targets,y_pred)
    





# Print average validation results across folds
print("Average Validation Loss:", np.mean(val_loss))
print("Average Validation MAE:", np.mean(val_mae))
print("Average Validation R2:", np.mean(Rsquare))
"""
import shap

explainer = shap.DeepExplainer(TL_model,train_data)
shap_values = explainer.shap_values(test_data,nsamples=150)
shap.summary_plot(shap_values,test_data,feature_names=column_names)
shap.summary_plot(shap_values[0], test_data,feature_names=column_names )
"""