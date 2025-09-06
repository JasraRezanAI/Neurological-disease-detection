
pip install streamlit scikit-learn pandas joblib
import streamlit as st
import joblib
import pandas as pd

# Add a title to your Streamlit application
st.title("Disease Detection Prediction")

# Add a brief description of the application
st.write("This application uses a trained model to predict disease detection based on input features.")

# Load the trained model and label encoder
try:
    best_model = joblib.load('best_model.joblib')
    le = joblib.load('label_encoder.joblib')
    st.success("Model and LabelEncoder loaded successfully.")
except FileNotFoundError:
    st.error("Model or LabelEncoder files not found. Please ensure 'best_model.joblib' and 'label_encoder.joblib' are in the same directory.")
    st.stop() # Stop the application if files are not found

# Create input fields for each feature
st.sidebar.header("Input Features")

# Define the expected features based on the training data (assuming X is available from previous steps)
# In a standalone app, you might load a sample data or schema to get feature names
# For this notebook context, we assume X is defined from previous cells.
# If running this script standalone, you would need to define feature names explicitly.
feature_names = ['Memory Recall (%)', 'Gait Speed (m/s)', 'Tremor Frequency (Hz)', 'Speech Rate (wpm)',
                 'Reaction Time (ms)', 'Eye Movement Irregularities (saccades/s)', 'Sleep Disturbance (scale 0-10)',
                 'Cognitive Test Score (MMSE)', 'Blood Pressure (mmHg)', 'Cholesterol (mg/dL)',
                 'Diabetes', 'Severity', 'Gender_Male']


feature_inputs = {}
for feature in feature_names:
    # Add appropriate input widgets based on feature type (using number_input for simplicity)
    # You might need different widgets for categorical or boolean features
    if feature in ['Sleep Disturbance (scale 0-10)', 'Diabetes', 'Severity', 'Gender_Male']:
        # Use number_input for these as they are treated as numerical in the model
        feature_inputs[feature] = st.sidebar.number_input(f"Enter value for {feature}", value=0, step=1)
    else:
        feature_inputs[feature] = st.sidebar.number_input(f"Enter value for {feature}", value=0.0)


# Convert input features to a pandas DataFrame
input_df = pd.DataFrame([feature_inputs])

st.subheader("Input Data")
st.write(input_df)

# Make predictions
if st.button("Predict"):
    try:
        # Ensure the input data has the same columns as the training data and in the correct order
        # This is crucial for the model to work correctly.
        input_df = input_df[feature_names]

        # Make prediction
        prediction_encoded = best_model.predict(input_df)

        # Decode the prediction
        prediction_decoded = le.inverse_transform(prediction_encoded)

        st.subheader("Prediction")
        st.write(f"The predicted disease detection is: **{prediction_decoded[0]}**")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
