import numpy as np
import pickle

# load the saved model
loaded_model = pickle.load(open('kidney_model_data.sav', 'rb'))

input_data = (71,0.3,40.9,0,1,46.8,1622,1) # No Dialysis Present)
input_data = (28,1.55,14.5,1,0,73.4,1096,0) # Dialysis Present

# change the input data into a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array for predicting 2 instances
input_data_reshaped = input_data_as_numpy_array(1,-1)

# make prediction
prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
    print("No Dialysis Present")
else:
    print("Dialysis Present")

if (prediction[1] == 1):
    print("Dialysis Present")
else:
    print("No Dialysis Present")