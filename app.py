# pip install -r requirements.txt
# streamlit run app.py


import streamlit as st
import pickle
import pandas as pd


def main():
    st.set_page_config(
    page_title="Insurance Prediction",
    page_icon="💸",
    # layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
    )
    st.title('Insurance Prediction')
    st.image('image.jpeg', width=700)
    st.write('In the section below, I will take you through the task of Insurance Prediction with Machine Learning using Python. For the task of Insurance prediction with machine learning, I have collected a dataset from Kaggle about the previous customers of a travel insurance company. Here our task is to train a machine learning model to predict whether an individual will purchase the insurance policy from the company or not.')    
    st.write('Let\'s start the task of Insurance prediction with machine learning by importing the necessary Python libraries and the dataset:')
    
    data = pd.read_csv('TravelInsurancePrediction.csv')
    st.expander("Show Data").dataframe(data,hide_index=True)

    code = '''import pandas as pd
data = pd.read_csv("TravelInsurancePrediction.csv")
data.head()'''
    st.code(code, language='python')



    age = st.number_input('Age', step=1, placeholder='Type here')
    option = st.selectbox('Employment Type', ('Government Sector', 'Private Sector/Self Employed'))
    employmentType = 0 if option == 'Government Sector' else 1
    annualIncome = st.number_input('Annual Income', step=1, placeholder='Type here')
    familyMembers = st.number_input('Family Members', step=1, placeholder='Type here')
    graduateOrNot = st.checkbox('Graduated')
    chronicDiseases = st.checkbox('Chronic Diseases')
    frequentFlyer = st.checkbox('Frequent Flyer')
    everTravelledAbroad = st.checkbox('Ever Travelled Abroad')

    features = [age, employmentType, graduateOrNot, annualIncome, familyMembers, chronicDiseases, frequentFlyer,
                everTravelledAbroad]

    model = pickle.load(open('model.sav', 'rb'))  # rb-->read binarey
    scaler = pickle.load(open('scaler.sav', 'rb'))

    btn = st.button('PREDICT')  # FOR PREDICTION BUTTON

    if btn:
        prediction = model.predict(scaler.transform([features]))
        if prediction == 0:
            st.write('## Will Buy INSURANCE')  # writ-->instead of print in st
        else:
            st.write('## Will Not Buy Insurance')


main()
