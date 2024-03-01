import streamlit as st
import pickle
from PIL import Image


def main():
    st.title('Insurance Prediction')
    image = Image.open('image.jpeg')
    st.image(image, width=700)
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
