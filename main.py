import streamlit as st
from pycaret.classification import * 
#import pickle


gbc_model = load_model('gbc')

st.title("Iris classifier")

a = float(st.number_input("sepal length in cm"))
b = float(st.number_input("sepal width in cm"))
c = float(st.number_input("petal length in cm"))
d = float(st.number_input("petal width in cm"))

inputs = [[a,b,c,d]]

input2 = [[5,4,1.7,0.4]]

pred = predict_model(gbc_model,data=input2)
print(pred)


#if st.button("Classify"):
  #st.success(classify((gbc_model.predict(inputs))))
  #result = predict
  #pass


