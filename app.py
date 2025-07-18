import streamlit as st
import pandas as pd
from joblib import load

# Load model
model = load("bestmodel_compressed.pkl")

# Encoding function
def encode_input(data):
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
st.title("👩‍💼 Employee Salary Class Predictor")

# Inputs
age = st.number_input("Age", 18, 100)
workclass = st.selectbox("Workclass", [
    'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
    'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'
])
fnlwgt = st.number_input("Fnlwgt", 10000, 1000000)
edu_num = st.number_input("Education Number", 1, 20)
marital = st.selectbox("Marital Status", [
    'Married-civ-spouse', 'Divorced', 'Never-married',
    'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'
])
occupation = st.selectbox("Occupation", [
    'Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial',
    'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical',
    'Farming-fishing', 'Transport-moving', 'Priv-house-serv',
    'Protective-serv', 'Armed-Forces'
])
relationship = st.selectbox("Relationship", [
    'Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'
])
race = st.selectbox("Race", [
    'White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'
])
gender = st.selectbox("Gender", ['Male', 'Female'])
cap_gain = st.number_input("Capital Gain", 0, 100000)
cap_loss = st.number_input("Capital Loss", 0, 100000)
hours = st.number_input("Hours per Week", 1, 100)
country = st.selectbox("Native Country", [
    'United-States', 'India', 'Mexico', 'Philippines', 'Germany',
    'Canada', 'England', 'China', 'Cuba', 'France', 'Other'
])

# Predict button
if st.button("Predict Salary Class"):
    # Create input DataFrame
    input_df = pd.DataFrame([{
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

    # Encode and predict
    input_df = encode_input(input_df).astype(float)

    try:
        prediction = model.predict(input_df)[0]
        st.success(f"📊 Predicted Salary Class: **{prediction}**")
    except Exception as e:
        st.error(f"🚨 Prediction Failed: {e}")
