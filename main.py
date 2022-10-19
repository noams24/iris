import streamlit as st
import pickle
import pandas as pd

def classify(data):
  if data == 0:
    return 'Iris-setosa'
  if data == 1:
    return 'Iris-versicolor'
  return 'Iris-virginica'


model = pickle.load(open('bestmodel.sav', 'rb'))

st.title("Iris classifier")

a = float(st.number_input("sepal length in cm"))
b = float(st.number_input("sepal width in cm"))
c = float(st.number_input("petal length in cm"))
d = float(st.number_input("petal width in cm"))

input_dict = {'PetalLengthCm' : a, 'PetalWidthCm' : b, 'SepalLengthCm' : c, 'SepalWidthCm' : d}
input_df = pd.DataFrame([input_dict])

if st.button("Classify"):
  st.success(classify(model.predict(input_df)[0]))
