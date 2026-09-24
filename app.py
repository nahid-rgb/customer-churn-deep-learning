import streamlit as st 
import numpy as np 
import tensorflow as tf 
import pandas as pd 
import pickle 
from tensorflow.keras.models import load_model 
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder 
 
 
# Load the trained deep learning model 
model  = load_model('model_artifacts/model.keras')  
 
# Load the Gender label encoder 
with open('model_artifacts/label_encoder_gender.pkl', mode='rb') as file: 
    label_encoder_gender = pickle.load(file) 
 
# Load the Geography one-hot encoder 
with open('model_artifacts/onehot_encoder_geo.pkl', mode='rb') as file: 
    onehot_encoder_geo = pickle.load(file)    
     
# Load the feature scaler 
with open('model_artifacts/scaler.pkl', mode='rb') as file: 
    scaler = pickle.load(file)     
 

# Streamlit app
st.title('🏦 Customer Churn Prediction')
st.write('Predict whether a customer is likely to leave the bank.')

# User input
st.subheader('👤 Customer Information')

geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
age = st.slider('🎂 Age', 18, 92)

st.subheader('💳 Banking Information')

balance = st.number_input('💰 Balance')
credit_score = st.number_input('📊 Credit Score')
estimated_salary = st.number_input('💵 Estimated Salary')
tenure = st.slider('📅 Tenure', 0, 10)
num_of_products = st.slider('🛍️ Number of Products', 1, 4)
has_cr_card = st.selectbox('💳 Has Credit Card', [0, 1])
is_active_member = st.selectbox('✅ Is Active Member', [0, 1])
 
# Prepare the input data  
input_data = pd.DataFrame({ 
   'CreditScore': [credit_score], 
    'Gender': [gender], 
    'Age': [age], 
    'Tenure': [tenure], 
    'Balance': [balance], 
    'NumOfProducts': [num_of_products], 
    'HasCrCard': [has_cr_card], 
    'IsActiveMember': [is_active_member], 
    'EstimatedSalary': [estimated_salary], 
    'Geography': [geography] 
}) 
 
# Encode the Gender input 
input_data['Gender'] = label_encoder_gender.transform(input_data['Gender']) 
 
 
# One-hot encode 'Geography' 
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray() 
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography'])) 
 
# Remove the original Geography column and add the one-hot encoded Geography columns 
df = pd.concat([input_data.drop('Geography', axis=1), geo_encoded_df], axis=1) 
 
# Scale the input data 
input_data_scaled = scaler.transform(df) 
 
 
# Predict churn 
prediction = model.predict(input_data_scaled) 
prediction_probabiliy = prediction[0][0] 
 
st.write(f'Churn Probability: {prediction_probabiliy:.2f}') 
 
if prediction_probabiliy > 0.5: 
    st.write('The customer is likely to churn.') 
else: 
    st.write('The customer is not likely to churn.')