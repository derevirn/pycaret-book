import pandas as pd
import streamlit as st
from pycaret.classification import load_model, predict_model

st.set_page_config(page_title="Iris Classification App")

@st.cache(allow_output_mutation=True)
def get_model():
    return load_model('classification_model')

def predict(model, df):
    predictions = predict_model(model, data = df)
    return predictions['prediction_label'][0]

model = get_model()


st.title("Iris Classification App")
st.markdown("Choose the values for each attribute of the Iris plant that you\
        want to be classified. This is a simple app showcasing the abilities\
        of the PyCaret classification module, based on Streamlit. For more\
        information visit the [Simplifying Machine Learning with PyCaret]\
        (https://leanpub.com/pycaretbook/) book website.")

form = st.form("species")
sepal_length = form.slider('Sepal Length', min_value=0.0, max_value=10.0, 
                           value=0.0, step = 0.1, format = '%f')
sepal_width = form.slider('Sepal Width', min_value=0.0, max_value=10.0,
                           value=0.0, step = 0.1, format = '%f')
petal_length = form.slider('Petal Length', min_value=0.0, max_value=10.0,
                           value=0.0, step = 0.1, format = '%f')
petal_width = form.slider('Petal Width', min_value=0.0, max_value=10.0,
                           value=0.0, step = 0.1, format = '%f')
    
predict_button = form.form_submit_button('Predict')

input_dict = {'sepal_length' : sepal_length, 'sepal_width' : sepal_width,
              'petal_length' : petal_length, 'petal_width' : petal_width}
            
input_df = pd.DataFrame([input_dict])

if predict_button:
    out = predict(model, input_df)
    st.success(f'The predicted species is {out}.')
    
