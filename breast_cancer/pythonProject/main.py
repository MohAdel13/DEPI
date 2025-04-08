import streamlit as st
import pandas as pd
import testing

st.title("🧬 Breast Cancer Record Entry")

st.markdown("Use the sliders below to enter a new record:")

# Define the numeric feature names (excluding 'id' and 'diagnosis')
features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
    'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
    'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
    'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
    'concavity_worst', 'concave points_worst', 'symmetry_worst',
    'fractal_dimension_worst'
]

# Create a dictionary to collect the input values
input_data = {}

for feature in features:
    input_data[feature] = st.slider(
        label=feature.replace("_", " ").capitalize(),
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=0.1
    )

# Combine into a DataFrame
if st.button("Add Record"):
    new_record = {
        **input_data
    }

    df_new = pd.DataFrame([new_record])
    st.success("✅ New record created!")
    st.dataframe(df_new)

    pred = testing.test(df_new)
    st.write("Prediction is:", pred)
