import streamlit as st
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import pickle

#loading models and necessary files

model = load_model('model.h5')
#geography encoder
with open('geography_encoder.pkl','rb') as file:
    geo_encoder = pickle.load(file)
#gender encoder
with open('gender_encoder.pkl','rb') as file:
    gender_encoder = pickle.load(file)
#scaler
with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

#streamlit app

st.title('Customer Churn Prediction')

geography = st.selectbox('Geography',geo_encoder.categories_[0])
gender = st.selectbox('Gender',gender_encoder.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Salary')
tenure = st.slider('Tenure',0,10)
number_of_product = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox("Has Credit Card",[0,1])
is_active = st.selectbox('Is active Member',[0,1])



test_df = {
    'CreditScore':[credit_score],
    'Gender':[gender_encoder.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[number_of_product],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active],
    'EstimatedSalary':[estimated_salary]
}
test_df = pd.DataFrame(test_df)
encoded_geo = pd.DataFrame(geo_encoder.transform([[geography]]).toarray(),columns = geo_encoder.get_feature_names_out())
final_df = pd.concat([test_df,encoded_geo],axis=1)
scaled_data = scaler.transform(final_df)
predict_proba = model.predict(scaled_data)[0][0]
st.write(f'Churn Probablity: {predict_proba:.2f}')
if predict_proba > 0.5:
    st.write('Customer likely to Churn')
else:
    st.write('Customer will not likely to Churn')



