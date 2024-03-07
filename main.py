# Importing ToolKits
import re
from time import sleep
import pandas as pd
import numpy as np

import streamlit as st
from streamlit.components.v1 import html
import warnings


def run():
    st.set_page_config(
        page_title="Heart Failure Detection",
        page_icon="❤",
        layout="wide"
    )

    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Function To Load Our Dataset
    @st.cache_data
    def load_model(model_path):
        return pd.read_pickle(model_path)

    model = pd.read_pickle("xgboost_heart_disease_detection_v1.pkl")

    st.markdown(
        """
    <style>
         .main {
            text-align: center;
         }
         h3{
            font-size: 25px
         }   
         .st-emotion-cache-16txtl3 h1 {
         font: bold 29px arial;
         text-align: center;
         margin-bottom: 15px

         }
         div[data-testid=stSidebarContent] {
         background-color: #111;
         border-right: 4px solid #222;
         padding: 8px!important

         }

         div.block-containers{
            padding-top: 0.5rem
         }

         .st-emotion-cache-z5fcl4{
            padding-top: 1rem;
            padding-bottom: 1rem;
            padding-left: 1.1rem;
            padding-right: 2.2rem;
            overflow-x: hidden;
         }

         .st-emotion-cache-16txtl3{
            padding: 2.7rem 0.6rem
         }

         .plot-container.plotly{
            border: 1px solid #333;
            border-radius: 6px;
         }

         div.st-emotion-cache-1r6slb0 span.st-emotion-cache-10trblm{
            font: bold 24px tahoma
         }
         div [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }

        div[data-baseweb=select]>div{
            cursor: pointer;
            background-color: #111;
            border: 4px solid #333
        }
        div[data-baseweb=select]>div:hover{
            border: 4px solid #B72F39

        }

        div[data-baseweb=base-input]{
            background-color: #111;
            border: 4px solid #444;
            border-radius: 5px;
            padding: 5px
        }

        div[data-testid=stFormSubmitButton]> button{
            width: 40%;
            background-color: #111;
            border: 2px solid #B72F39;
            padding: 18px;
            border-radius: 30px;
            opacity: 0.8;
        }
        div[data-testid=stFormSubmitButton]  p{
            font-weight: bold;
            font-size : 20px
        }

        div[data-testid=stFormSubmitButton]> button:hover{
            opacity: 1;
            border: 2px solid #B72F39;
            color: #fff
        }

    </style>
    """,
        unsafe_allow_html=True
    )

    header = st.container()
    content = st.container()

    st.write("")

    with header:
        st.title("Heart Failure Prediction 💔")
        st.write("")

    with content:
        col1, col2 = st.columns([7, 5])

        with col1:
            with st.form("Preidct"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    age = st.number_input('Age', min_value=1,
                                          max_value=90, value=48)

                    max_heart_rate = st.number_input('Max Heart Rate', min_value=0,
                                                     max_value=200, value=100)

                    ecg = st.selectbox('ECG', options=[
                        "Normal", "ST", "LVH"], index=0)

                    st_slope = st.selectbox('ST Slope', options=[
                                            "Up", "Flat", "Down"], index=0)

                with c2:
                    blood_pressure = st.number_input('Resting Blood Pressure', min_value=0,
                                                     max_value=200, value=140)

                    old_peak = st.number_input('Old Peak', min_value=-3.0,
                                               max_value=4.5, value=2.5)

                    chest_pain_type = st.selectbox('Chest Pain Type', options=[
                        "Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"], index=0)

                    exercise_angina = st.selectbox(
                        'Exercise Angina', options=["No", "Yes"], index=0)
                with c3:
                    cholesterol = st.number_input('Cholesterol', min_value=0,
                                                  max_value=510, value=228)

                    st.write("")

                    gender = st.selectbox('Gender', options=[
                        "Male", "Female"], index=0)

                    fasting_blood_sugar = st.selectbox('Fasting Blood Sugar', options=[
                        "Greater Than 120 mg/dl", "Less Than 120 mg/dl"], index=0)

                predict_button = st.form_submit_button("Predict 🚀")

        with col2:
            if predict_button:
                patient_fasting_blood_sugar = 1
                if fasting_blood_sugar == "Less Than 120 mg/dl":
                    patient_fasting_blood_sugar = 0

                new_data = [age, blood_pressure, cholesterol,
                            patient_fasting_blood_sugar, max_heart_rate, old_peak]

                # Gender
                patient_gender = [1]  # Male

                if gender == "Female":
                    patient_gender = [0]  # Female

                # Chest Pain
                patient_chest_pain_type = [0, 0, 0]  # ASY

                if chest_pain_type == "Typical Angina":
                    patient_chest_pain_type = [0, 0, 1]

                elif chest_pain_type == "Atypical Angina":
                    patient_chest_pain_type = [1, 0, 0]

                elif chest_pain_type == "Non-anginal Pain":
                    patient_chest_pain_type = [0, 1, 0]

                # ECG
                patinet_ecg = [0, 0]  # LVH

                if ecg == "Normal":
                    patinet_ecg = [1, 0]

                elif ecg == "ST":
                    patinet_ecg = [0, 1]

                # ExerciseAngina
                patient_exercise_angina = [1]  # Yes

                if exercise_angina == "No":
                    patient_exercise_angina = [0]  # No

                # Slope
                patient_slope = [0, 0]  # Down
                if st_slope == "Flat":
                    patient_slope = [1, 0]
                elif st_slope == "Up":
                    patient_slope = [0, 1]

                # Appending All Data
                new_data.extend(patient_gender)
                new_data.extend(patient_chest_pain_type)
                new_data.extend(patinet_ecg)
                new_data.extend(patient_exercise_angina)
                new_data.extend(patient_slope)

                with st.spinner(text='Predict The Value..'):

                    predicted_value = model.predict([new_data])[0]
                    prediction_prop = np.round(
                        model.predict_proba([new_data])*100)
                    sleep(1.2)

                    heart_disease, no_heart_disease = st.columns(2)

                    st.image("imgs/heartbeat.png",
                             caption="", width=100)
                    if predicted_value == 0:
                        st.subheader("Expected He Is")
                        st.subheader(":green[Not a Heart Patient]")

                    else:
                        st.subheader(f"Expected He Is")
                        st.subheader(":red[Heart Patient]")

                    with heart_disease:
                        st.image("imgs/heart.png", caption="", width=65)
                        st.subheader(":green[*Not Heart Patient*]")
                        st.subheader(f"{prediction_prop[0, 0]}%")

                    with no_heart_disease:
                        st.image("imgs/hearted.png", caption="", width=65)
                        st.subheader(f":red[*Heart Patient*]")
                        st.subheader(f"{prediction_prop[0, 1]}%")


run()
