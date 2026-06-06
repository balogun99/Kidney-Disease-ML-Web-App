import numpy as np
import pickle
import streamlit as st

# load the saved model
loaded_model = pickle.load(open('kidney_model_data.sav', 'rb'))

# with open('/Users/macbookpro/Desktop/ML & DL & GENAI Projects/Machine Learning/Kidney Disease Predictive System/kidney_model_data.sav', 'rb') as file:
#     loaded_model = pickle.load(file)

# create a function for prediction
def kidney_disease_prediction(input_data):
    # change the input array into a numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    # reshape the array for predicting 2 instances
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return "No Dialysis Present"
    else:
        return "Dialysis Present"
    
# Streamlit Interface
def main():
    # web title
    st.title("Kidney Disease Prediction Web App")
    # input from the user
    Age = st.text_input("Age of an Individual")
    Creatinine_Level = st.text_input("Creatinine Level")
    BUN = st.text_input("BUN Level")
    Diabetes = st.text_input("Diabetes Status")
    Hypertension = st.text_input("Hypertension Status")
    GFR = st.text_input("GFR Level")
    Urine_Output = st.text_input("Urine Output Status")
    CKD_Status = st.text_input("CKD Status")

    diagnosis = ''

    if st.button("Kidney Disease Prediction"):
        diagnosis = kidney_disease_prediction([Age,Creatinine_Level,BUN,
                                               Diabetes,Hypertension,GFR,
                                               Urine_Output,CKD_Status])
        st.success(diagnosis)

if __name__ == '__main__':
    main()