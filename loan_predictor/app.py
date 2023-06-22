import streamlit as st
import numpy as np
import pandas as pd
from data import professions_list
from data import features
from sklearn.preprocessing import MinMaxScaler
import pickle as pkl

st.set_page_config(page_title='Loan Prediction')

st.title("Loan Prediction")

income = st.number_input('Enter your Income', value=25000)

age = st.slider('How old are you?', 0, 100, 25)
st.write("I'm ", age, 'years old')

maritalStatus = st.radio(
    "What\'s your Marital Status ?",
    ('Single', 'Married'))

selected_profession = st.selectbox(
    'What is your Profession ?',
    list(professions_list.keys()))

experience = st.slider(
    "What is your experience of your profession?", 0, 50, 10)

car_ownership = st.radio(
    "Do you have Car Ownership ? ",
    ('Yes', 'No'))

house_ownership = st.radio(
    "What's your House Ownership ? ",
    ('Rented', 'No Rented, No Own', 'Own'))

jobYears = st.slider(
    "What is your current job years?", 0, 50, 10)

houseYears = st.slider(
    "What is your current House years?", 0, 50, 10)

# Preprocessing

# Marital status
if maritalStatus == 'Single':
    maritalStatus = 1
else:
    maritalStatus = 0

# House Ownership
if house_ownership == 'Rented':
    house_ownership = 2
elif house_ownership == 'No Rented, No Own':
    house_ownership = 0
else:
    house_ownership = 1

# Profession
profession = professions_list[selected_profession]

# car ownership
if car_ownership == 'No':
    car_ownership = 0
else:
    car_ownership = 1


userData = [[income, age, experience, maritalStatus,
             house_ownership, car_ownership, profession, jobYears, houseYears]]

df = pd.DataFrame(userData, columns=features)

model = pkl.load(open('./loan_pred_model.pkl', 'rb'))

if st.button("PREDICT"):
    pred = model.predict(df)
    if pred == 1:
        st.subheader("Granting Loan to the Customer is Risky")
    else:
        st.subheader("Congrats The customer is eligible for Loan")
        st.balloons()