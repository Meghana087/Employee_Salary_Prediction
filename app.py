import streamlit as st
import pandas as pd
import joblib
from joblib import load
import numpy as np

# Load the trained and compressed model
model = load("bestmodel_compressed.pkl")

# Define label encodings for categorical features (used during training)
def encode_input(data):
    # You should modify these mappings to exactly match your training pipeline
    mapping = {
        'workclass': {
            'Private': 0, 'Self-emp-not-inc': 1, 'Self-emp-inc': 2, 'Federal-gov': 3,
            'Local-gov': 4, 'State-gov': 5, 'Without-pay': 6, 'Never-worked': 7
        },
        'marital-status': {
            'Married-civ-spouse': 0, 'Divorced': 1, 'Never-married': 2,
            'Separated': 3, 'Widowed': 4, 'Married-spouse-absent': 5, 'Married-AF-spouse': 6
        },
        'occupation': {
            'Tech-support': 0, 'Craft-repair': 1, 'Other-service': 2, 'Sales': 3,
            'Exec-managerial': 4, 'Prof-specialty': 5, 'Handlers-cleaners': 6,
            'Machine-op-inspct': 7, 'Adm-clerical': 8, 'Farming-fishing': 9,
            'Transport-moving': 10, 'Priv-house-serv': 11, 'Protective-serv': 12,
            'Armed-Forces': 13
        },
        'relationship': {
            'Wife': 0, 'Own-child': 1, 'Husband': 2, 'Not-in-family': 3,
            'Other-relative': 4, 'Unmarried': 5
        },
        'race': {
            'White': 0, 'Black': 1, 'Asian-Pac-Islander': 2,
            'Amer-Indian-Eskimo': 3, 'Other': 4
        },
        'gender': {'Male': 0, 'Female': 1},
        'native-country': {
            'United-States': 0, 'India': 1, 'Mexico': 2, 'Philippines': 3,
            'Germany': 4, 'Canada': 5, 'England': 6, 'China': 7,
            'Cuba': 8, 'France': 9, 'Other': 10
        }
    }

    for col in mapping:
        data[col] = data[col].map(mapping[col]).fillna(-1)

    return data

# Streamlit UI
st.title("👨‍💼 Employee Salary Class Predictor")
st.markdown("Enter employee details below to predict their income class:")

# Input fields
age = st.number_input("Age", 18, 100)
workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
fnlwgt = st.number_input("Fnlwgt", 10000, 1000000)
edu_num = st.number_input("Education Num", 1, 20)
marital = st.selectbox("Marital Status", ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
occupation = st.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty',
                                         'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving',
                                         'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])
relationship = st.selectbox("Relationship", ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
race = st.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
gender = st.selectbox("Gender", ['Male', 'Female'])
cap_gain = st.number_input("Capital Gain", 0, 100000)
cap_loss = st.number_input("Capital Loss", 0, 100000)
hours = st.number_input("Hours per Week", 1, 100)
country = st.selectbox("Native Country", ['United-States', 'India', 'Mexico', 'Philippines', 'Germany', 'Canada', 'England', 'China', 'Cuba', 'France', 'Other'])

# Predict
if st.button("Predict"):
    user_input = pd.DataFrame([{
        'age': age,
        'workclass': workclass,
        'fnlwgt': fnlwgt,
        'educational-num': edu_num,
        'marital-status': marital,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'gender': gender,
        'capital-gain': cap_gain,
        'capital-loss': cap_loss,
        'hours-per-week': hours,
        'native-country': country
    }])

    st.write("🔎 Raw Input:", user_input)

    # Encode categorical features
    user_input = encode_input(user_input)

    # Ensure float type
    user_input = user_input.astype(float)

    # Debug info
    st.write("✅ Processed Input:", user_input)
    st.write("Input Shape:", user_input.shape)
    st.write("Any Nulls?", user_input.isnull().sum())

    try:
        prediction = model.predict(user_input)[0]
        st.success(f"📊 Predicted Salary Class: **{prediction}**")
    except Exception as e:
        st.error(f"🚨 Prediction Failed: {e}")
