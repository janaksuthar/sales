import pandas as pd
import numpy as np
import joblib
import streamlit as st

#Load the ML model

model=joblib.load(open("linear_regression_model.joblib", 'rb'))

st.title("Sales Prediction app")

# Input feature 

TV=st.number_input("TV Adv budget", min_value=0.0)
Radio=st.number_input("Radio Adv budget", min_value=0.0)
Newspaper=st.number_input("Newspaper Adv budget", min_value=0.0)

#Make Pred

if st.button('predict sales'):
	input_data=np.array([[TV, Radio,Newspaper]])
	prediction=model.predict(input_data)[0]

	st.sucess(f'predict sales:{prediction:.2f}')
