import joblib
import streamlit as st
import numpy as np

FILE = "./regression.joblib"

model = joblib.load(FILE)

st.title("House Price Prediction")
size = st.number_input("Insert a size")
bedrooms = st.number_input("Insert a number of bedrooms")
garden = st.checkbox("Does the house have a garden")

X_test = np.array([[size, bedrooms, garden]])

predict = model.predict(X_test)

st.write(f"The predicted price is {predict[0]:.2f}k$")
