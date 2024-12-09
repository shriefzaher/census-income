import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder


# Load the pre-trained Random Forest model
rf_model = joblib.load('./assets/random_forest_model.pkl')
LR_model = joblib.load('./assets/LR_model_model.pkl')
xg_model = joblib.load('./assets/XG_model_model.pkl')

# Streamlit app
st.title("Prediction App")



# Initialize all possible one-hot encoded columns to 0
input_data = {
    'age_Adult': 0, 'age_Old': 0,
    'workclass_Local-gov': 0, 'workclass_Never-worked': 0, 'workclass_Private': 0, 
    'workclass_Self-emp-inc': 0, 'workclass_Self-emp-not-inc': 0, 'workclass_State-gov': 0, 
    'workclass_Without-pay': 0, 
    'marital_status_Married-AF-spouse': 0, 'marital_status_Married-civ-spouse': 0,
    'marital_status_Married-spouse-absent': 0, 'marital_status_Never-married': 0,
    'marital_status_Separated': 0, 'marital_status_Widowed': 0,
    'occupation_Armed-Forces': 0, 'occupation_Craft-repair': 0, 'occupation_Exec-managerial': 0, 
    'occupation_Farming-fishing': 0, 'occupation_Handlers-cleaners': 0, 'occupation_Machine-op-inspct': 0,
    'occupation_Other-service': 0, 'occupation_Priv-house-serv': 0, 'occupation_Prof-specialty': 0,
    'occupation_Protective-serv': 0, 'occupation_Sales': 0, 'occupation_Tech-support': 0,
    'occupation_Transport-moving': 0, 
    'relationship_Not-in-family': 0, 'relationship_Other-relative': 0, 'relationship_Own-child': 0,
    'relationship_Unmarried': 0, 'relationship_Wife': 0,
    'race_White': 0, 'sex_Male': 0, 'native_country_United-States': 0,
    'employment_type_Normal Hours': 0, 'employment_type_Extra Hours': 0,
    'education_Assoc-acdm': 0, 'education_Assoc-voc': 0, 'education_Bachelors': 0, 
    'education_Doctorate': 0, 'education_HS-grad': 0, 'education_Masters': 0, 
    'education_Prof-school': 0, 'education_Some-college': 0,
    'Capital_Difference_Major': 0
}

# Define mappings for each input category
age_mapping = {
    'Young': {'age_Adult': 0, 'age_Old': 0},
    'Adult': {'age_Adult': 1, 'age_Old': 0},
    'Old': {'age_Adult': 0, 'age_Old': 1}
}

workclass_mapping = {
    'Local-gov': {'workclass_Local-gov': 1}, 'Never-worked': {'workclass_Never-worked': 1},
    'Private': {'workclass_Private': 1}, 'Self-emp-inc': {'workclass_Self-emp-inc': 1},
    'Self-emp-not-inc': {'workclass_Self-emp-not-inc': 1}, 'State-gov': {'workclass_State-gov': 1},
    'Without-pay': {'workclass_Without-pay': 1}
}

marital_status_mapping = {
    'Married-AF-spouse': {'marital_status_Married-AF-spouse': 1},
    'Married-civ-spouse': {'marital_status_Married-civ-spouse': 1},
    'Married-spouse-absent': {'marital_status_Married-spouse-absent': 1},
    'Never-married': {'marital_status_Never-married': 1},
    'Separated': {'marital_status_Separated': 1},
    'Widowed': {'marital_status_Widowed': 1}
}

occupation_mapping = {
    'Armed-Forces': {'occupation_Armed-Forces': 1},
    'Craft-repair': {'occupation_Craft-repair': 1},
    'Exec-managerial': {'occupation_Exec-managerial': 1},
    'Farming-fishing': {'occupation_Farming-fishing': 1},
    'Handlers-cleaners': {'occupation_Handlers-cleaners': 1},
    'Machine-op-inspct': {'occupation_Machine-op-inspct': 1},
    'Other-service': {'occupation_Other-service': 1},
    'Priv-house-serv': {'occupation_Priv-house-serv': 1},
    'Prof-specialty': {'occupation_Prof-specialty': 1},
    'Protective-serv': {'occupation_Protective-serv': 1},
    'Sales': {'occupation_Sales': 1},
    'Tech-support': {'occupation_Tech-support': 1},
    'Transport-moving': {'occupation_Transport-moving': 1}
}

relationship_mapping = {
    'Not-in-family': {'relationship_Not-in-family': 1},
    'Other-relative': {'relationship_Other-relative': 1},
    'Own-child': {'relationship_Own-child': 1},
    'Unmarried': {'relationship_Unmarried': 1},
    'Wife': {'relationship_Wife': 1}
}


race_mapping = {
    'White': {'race_White': 1},
    'Others': {}
    # Add more race categories here if needed
}

sex_mapping = {
    'Male': {'sex_Male': 1},
    'Female': {}  # No need to set anything for female if Male is 0 by default
}

native_country_mapping = {
    'United States': {'native_country_United-States': 1},
    'Others': {}
    # Add more countries if needed
}

employment_type_mapping = {
    'Normal Hours': {'employment_type_Normal Hours': 1},
    'Extra Hours': {'employment_type_Extra Hours': 1}
}

education_mapping = {
    'Assoc-acdm': {'education_Assoc-acdm': 1},
    'Assoc-voc': {'education_Assoc-voc': 1},
    'Bachelors': {'education_Bachelors': 1},
    'Doctorate': {'education_Doctorate': 1},
    'HS-grad': {'education_HS-grad': 1},
    'Masters': {'education_Masters': 1},
    'Prof-school': {'education_Prof-school': 1},
    'Some-college': {'education_Some-college': 1}
}

Capital_Difference_mapping = {
    'Major': {'Capital_Difference_Major': 1},
    'Minor': {}  # No need to set anything for female if Male is 0 by default
}

############################## UI ##############################

# Sample user inputs (you'd get these from `st.selectbox()` in Streamlit)
age = st.selectbox("age", options=["Young", "Adult", "Old"])
workclass = st.selectbox("workclass", options=['Local-gov', 'Never-worked', 'Private', 
                                               'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay'])
marital_status = st.selectbox("marital status", options=['Married-AF-spouse', 'Married-civilian-spouse', 
                                                         'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'])
occupation = st.selectbox("occupation", options=['Armed-Forces', 'Craft-repair', 'Exec-managerial', 'Farming-fishing', 
                                                 'Handlers-cleaners', 'Machine-operation-inspct', 'Other-service', 
                                                 'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 
                                                 'Sales', 'Tech-support', 'Transport-moving'])
relationship = st.selectbox("relationship", options=['Not-in-family', 'Other-relative', 'Own-child', 'Unmarried', 'Wife'])
race = st.selectbox("race", options=['White','Others'])
sex = st.selectbox("sex", options=['Male', 'Female'])
native_country = st.selectbox("native country", options=['United States','Others'])
employment_type = st.selectbox("employment type", options=['Normal Hours', 'Extra Hours'])
education = st.selectbox("education", options=['Assoc-acdm', 'Assoc-voc', 'Bachelors', 'Doctorate', 'HS-grad', 'Masters', 
                                               'Prof-school', 'Some-college'])
Capital_Difference = st.selectbox("Capital Difference", options=['Major','Minor'])


# Updating input_data based on user selections
input_data.update(age_mapping[age])
input_data.update(workclass_mapping[workclass])
input_data.update(marital_status_mapping[marital_status])
input_data.update(occupation_mapping[occupation])
input_data.update(relationship_mapping[relationship])
input_data.update(race_mapping[race])
input_data.update(sex_mapping[sex])
input_data.update(native_country_mapping[native_country])
input_data.update(employment_type_mapping[employment_type])
input_data.update(education_mapping[education])
input_data.update(Capital_Difference_mapping[Capital_Difference])

input_df = pd.DataFrame([input_data])




##############################  Random Forest Prediction     ##############################

# When the user clicks 'Predict'
if st.button("Random Forest Prediction"):
    # Predict using the model
    # print("Input data",input_data)

    prediction = rf_model.predict(input_df)
    prediction_proba = rf_model.predict_proba(input_df)

    # Map prediction to target names
    Income = ['less than 50K', 'more than 50K']
    predicted_income = Income[prediction[0]]
    max_proba = max(prediction_proba[0]) 
    
    # Display the result
    st.subheader(f"(Random Forest) Predicted Income is: {predicted_income}")
    st.subheader(f"(Random Forest) Prediction confidence: {max_proba*100:.2f} %")


############################## XG Prediction     ##############################

# When the user clicks 'Predict'
if st.button("XG Prediction"):
    # Predict using the model
    # print("Input data",input_data)

    prediction = xg_model.predict(input_df)
    prediction_proba = xg_model.predict_proba(input_df)

    # Map prediction to target names
    Income = ['less than 50K', 'more than 50K']
    predicted_income = Income[prediction[0]]
    max_proba = max(prediction_proba[0]) 
    
    # Display the result
    st.subheader(f"(XG) Predicted Income is: {predicted_income}")
    st.subheader(f"(XG) Prediction confidence: {max_proba*100:.2f} %")




############################## LR Prediction     ##############################

# When the user clicks 'Predict'
if st.button("LR Prediction"):
    # Predict using the model
    # print("Input data",input_data)

    prediction = LR_model.predict(input_df)
    prediction_proba = LR_model.predict_proba(input_df)

    # Map prediction to target names
    Income = ['less than 50K', 'more than 50K']
    predicted_income = Income[prediction[0]]
    max_proba = max(prediction_proba[0]) 
    
    # Display the result
    st.subheader(f"(LR) Predicted Income is: {predicted_income}")
    st.subheader(f"(LR) Prediction confidence: {max_proba*100:.2f} %")



