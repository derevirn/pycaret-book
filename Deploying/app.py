import numpy as np
import pandas as pd
import streamlit as st
from pycaret.regression import load_model, predict_model

@st.cache(allow_output_mutation=True)
def get_model():
    return load_model('regression_model')

def predict(model, df):
    predictions = predict_model(model, data = df)
    return predictions['Label'][0]

model = get_model()

st.title("Insurance Charges Prediction App")

form = st.form("charges")
age = form.number_input('Age', min_value=1, max_value=100, value=25)
sex = form.selectbox('Sex', ['male', 'female'])
bmi = form.number_input('BMI', min_value=10, max_value=50, value=10)
children = form.number_input('Children', min_value=0, max_value=10, value=0)

if form.checkbox('Smoker'):
    smoker = 'yes'
else:
    smoker = 'no'

region_list = ['southwest', 'northwest', 'northeast', 'southeast']
region = form.selectbox('Region', region_list)
predict_button = form.form_submit_button('Predict')


input_dict = {'age' : age, 'sex' : sex, 'bmi' : bmi, 'children' : children,
              'smoker' : smoker, 'region' : region}
input_df = pd.DataFrame([input_dict])

if predict_button:
    out = predict(model, input_df)
    st.success('The predicted charges are ${:.2f}'.format(out))