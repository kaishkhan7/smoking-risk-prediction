import streamlit as st
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model(
    "model/smoking_risk_model.keras"
)

# Load scaler values
scaler_mean = np.load(
    "model/scaler_mean.npy"
)

scaler_scale = np.load(
    "model/scaler_scale.npy"
)


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="Kaish Khan's Smoking Risk Prediction deep learning Project",
    page_icon="🫁",
    layout="centered"
)


# -----------------------------
# Title
# -----------------------------

st.title("🫁 Smoking Health Risk AI")
st.write(
    "Kaish Khan's Smoking Risk Prediction deep learning Project "
    "Educational AI-based smoking health risk assessment"
)

st.warning(
    "⚠️ This is an educational prediction tool, "
    "not a medical diagnosis."
)


# -----------------------------
# User information
# -----------------------------

st.header("👤 Personal Information")

name = st.text_input(
    "Enter your name"
)

age = st.number_input(
    "Enter your age",
    min_value=18,
    max_value=100,
    value=19
)


# -----------------------------
# Smoking information
# -----------------------------

st.header("🚬 Smoking Information")

smoker_option = st.radio(
    "Do you smoke?",
    ["No", "Yes"]
)

if smoker_option == "Yes":

    smoker = 1

    smoking_years = st.number_input(
        "How many years have you been smoking?",
        min_value=1,
        max_value=80,
        value=1
    )

    cigarettes = st.number_input(
        "How many cigarettes per day?",
        min_value=1,
        max_value=100,
        value=5
    )

else:

    smoker = 0
    smoking_years = 0
    cigarettes = 0


# -----------------------------
# Symptoms
# -----------------------------

st.header("🩺 Symptoms")

cough = st.checkbox(
    "Frequent cough"
)

breathing_problem = st.checkbox(
    "Breathing problem"
)

chest_pain = st.checkbox(
    "Chest pain"
)


# Convert symptoms to 0/1

cough_value = 1 if cough else 0

breathing_value = 1 if breathing_problem else 0

chest_value = 1 if chest_pain else 0


# -----------------------------
# Prediction button
# -----------------------------

if st.button(
    "🔍 Predict Health Risk",
    use_container_width=True
):

    if name.strip() == "":
        st.error("Please enter your name.")

    else:

        # Create input array
        input_data = np.array([
            [
                age,
                smoker,
                smoking_years,
                cigarettes,
                cough_value,
                breathing_value,
                chest_value
            ]
        ], dtype=float)


        # Apply same scaling used during training
        input_scaled = (
            input_data - scaler_mean
        ) / scaler_scale


        # Prediction
        prediction = model.predict(
            input_scaled,
            verbose=0
        )[0]


        predicted_class = np.argmax(
            prediction
        )


        confidence = (
            prediction[predicted_class] * 100
        )


        # -----------------------------
        # Result
        # -----------------------------

        st.header("📊 Prediction Result")

        st.write(
            f"### Hello, {name}!"
        )


        if predicted_class == 0:

            st.success(
                "🟢 LOW RISK"
            )

            risk_text = (
                "The model estimates a lower "
                "smoking-related risk based on "
                "the information provided."
            )

        elif predicted_class == 1:

            st.warning(
                "🟡 MODERATE RISK"
            )

            risk_text = (
                "The model estimates a moderate "
                "smoking-related risk based on "
                "the information provided."
            )

        else:

            st.error(
                "🔴 HIGH RISK"
            )

            risk_text = (
                "The model estimates a higher "
                "smoking-related risk based on "
                "the information provided."
            )


        st.write(risk_text)

        st.write(
            f"**Model confidence: {confidence:.2f}%**"
        )


        # -----------------------------
        # Health information
        # -----------------------------

        st.header(
            "🫁 Smoking-related Health Information"
        )

        st.write(
            "Smoking is associated with increased "
            "risk of several serious health problems."
        )

        st.markdown("""
        - 🫁 Lung and respiratory diseases
        - ❤️ Heart and blood-vessel diseases
        - 🧠 Stroke
        - Cancer affecting several parts of the body
        - Reduced lung function
        - Long-term breathing problems
        """)


        st.info(
            "If you have concerning symptoms, talk to "
            "a qualified healthcare professional. "
            "This application cannot diagnose disease."
        )


# -----------------------------
# Footer
# -----------------------------

st.markdown("---")

st.caption(
    "Smoking Health Risk AI • Deep Learning Project"
)