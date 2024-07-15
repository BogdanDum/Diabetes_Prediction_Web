#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 13:00:10 2024

@author: bogdanduminica
"""

import numpy as np
import pickle
import streamlit as st

# load the model
loaded_model = pickle.load(open("/Users/bogdanduminica/Desktop/Diabetes_Prediction_Project/trained_model.sav", "rb"))

# create a function for the actual prediction

def diabetes_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  

def main():
    
    # give title 
    st.title("Diabetes Prediction Web App")
    
    # get input data fron the user
    
    Pregnancies = st.text_input("Number of pregnancies")
    Glucose = st.text_input("Glucose level")
    BloodPressure = st.text_input("Blood pressure level")
    SkinThickness = st.text_input("Level of skin thickness")
    Insulin = st.text_input("Insulin levels")
    BMI = st.text_input("BMI index")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function value")
    Age = st.text_input("Age")
    
    # code for prediction
    diagnosis = ""
    
    # make a button for the actual prediction
    if st.button("Diabetes test result"):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
    
    st.success(diagnosis)
    
    
if __name__ == "__main__":
    main()
    
    