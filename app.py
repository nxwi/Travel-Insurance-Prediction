# pip install -r requirements.txt
# streamlit run app.py


import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
import pickle


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def main():
    st.set_page_config(
    page_title="Insurance Prediction",
    page_icon="💸",
    # layout="wide",
    initial_sidebar_state="expanded",
    # menu_items={
    #     'Get Help': 'https://www.extremelycoolapp.com/help',
    #     'Report a bug': "https://www.extremelycoolapp.com/bug",
    #     'About': "# This is a header. This is an *extremely* cool app!"
    # }
    )
    st.title('Insurance Prediction')
    st.image('image.jpeg', width=700)
    st.write('In the section below, I will take you through the task of Insurance Prediction with Machine Learning using Python. For the task of Insurance prediction with machine learning, I have collected a dataset from Kaggle about the previous customers of a travel insurance company. Here our task is to train a machine learning model to predict whether an individual will purchase the insurance policy from the company or not.')    
    
    col1, col2 = st.columns([1,7])

    with col1:
        st.link_button("Colab", "https://colab.research.google.com/drive/1quta7tpMjplPrpYfQ_0ew7FCNZgjqO_y?usp=sharing")

    with col2:
        st.link_button("DataSet", "https://raw.githubusercontent.com/amankharwal/Website-data/master/TravelInsurancePrediction.csv")
    
    tab01, tab02, = st.tabs(["Data", "Model"])

    with tab01:

        st.write('- Let\'s start the task of Insurance prediction with machine learning by importing the necessary Python libraries and the dataset:')
        st.code('''import pandas as pd
    data = pd.read_csv("TravelInsurancePrediction.csv")
    data.head()''', language='python')
        df = pd.read_csv('TravelInsurancePrediction.csv')
        st.dataframe(df, hide_index=True)

        st.write('- The `Unnamed` column in this dataset is of no use, so I\'ll just remove it from the data:')
        st.code('''df.drop(['Unnamed: 0'],axis=1,inplace=True)
    df''', language='python')
        df.drop(['Unnamed: 0'],axis=1,inplace=True)
        btn = st.button("Run", key=2)
        if btn:
            st.dataframe(df, hide_index=True)

        st.write('- Now let\'s look at some of the necessary insights to get an idea about what kind of data we are working with:')
        st.code('''df.isna().sum()''', language='python')
        btn = st.button("Run", key=3)
        if btn:
            st.dataframe(df.isna().sum())
        st.code('''df.dtypes''', language='python')
        btn = st.button("Run", key=4)
        if btn:
            st.dataframe(df.dtypes)

        st.write('- In this dataset, the labels we want to predict are in the “TravelInsurance” column. The values in this column are mentioned as 0 and 1 where 0 means not bought and 1 means bought. For a better understanding when analyzing this data, I will convert 1 and 0 to purchased and not purchased:')
        st.code('''df["TravelInsurance"] = df["TravelInsurance"].map({0: "Not Purchased", 1: "Purchased"})''', language='python')
        df["TravelInsurance"] = df["TravelInsurance"].map({0: "Not Purchased", 1: "Purchased"})
        btn = st.button("Run", key=5)
        if btn:
            st.dataframe(df, hide_index=True)

        st.write('- Now let\'s start by looking at the columns to see how they affects the purchase of an insurance policy:')

        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Age", "Employment Type", "Graduate Or Not","Annual Income", "Family Members", "Chronic Diseases", "Frequent Flyer", "Ever Travelled Abroad"])

        with tab1:
            st.code('''figure = px.histogram(df, x = "Age",color = "TravelInsurance")\nfigure.show()''', language='python')
            figure = px.histogram(df, x = "Age",color = "TravelInsurance")
            st.write(figure)
        with tab2:
            st.code('''figure = px.histogram(df, x = "Employment Type",color = "TravelInsurance")\nfigure.show()''', language='python')
            figure = px.histogram(df, x = "Employment Type",color = "TravelInsurance")
            st.write(figure)
        with tab3:
            st.code('''figure = px.histogram(df, x = "GraduateOrNot",color = "TravelInsurance")\nfigure.show()''', language='python')
            figure = px.histogram(df, x = "GraduateOrNot",color = "TravelInsurance")
            st.write(figure)
        with tab4:
            st.code('''figure = px.histogram(df, x = "AnnualIncome",color = "TravelInsurance")\nfigure.show()''', language='python')
            figure = px.histogram(df, x = "AnnualIncome",color = "TravelInsurance")
            st.write(figure)
        with tab5:
            st.code('''figure = px.histogram(df, x = "FamilyMembers",color = "TravelInsurance")\nfigure.show()''', language='python')
            figure = px.histogram(df, x = "FamilyMembers",color = "TravelInsurance")
            st.write(figure)
        with tab6:
            st.code('''figure = px.histogram(df, x = "ChronicDiseases",color = "TravelInsurance")\nfigure.show()''', language='python')
            figure = px.histogram(df, x = "ChronicDiseases",color = "TravelInsurance")
            st.write(figure)
        with tab7:
            st.code('''figure = px.histogram(df, x = "FrequentFlyer",color = "TravelInsurance")\nfigure.show()''', language='python')
            figure = px.histogram(df, x = "FrequentFlyer",color = "TravelInsurance")
            st.write(figure)
        with tab8:
            st.code('''figure = px.histogram(df, x = "EverTravelledAbroad",color = "TravelInsurance")\nfigure.show()''', language='python')
            figure = px.histogram(df, x = "EverTravelledAbroad",color = "TravelInsurance")
            st.write(figure)

        st.write('- I will convert all categorical values to 1 and 0 first because all columns are important for training the insurance prediction model:')
        st.code('''df["Employment Type"] = df["Employment Type"].map({"Government Sector": 0, "Private Sector/Self Employed": 1})
df["GraduateOrNot"] = df["GraduateOrNot"].map({"No": 0, "Yes": 1})
df["FrequentFlyer"] = df["FrequentFlyer"].map({"No": 0, "Yes": 1})
df["EverTravelledAbroad"] = df["EverTravelledAbroad"].map({"No": 0, "Yes": 1})
df["TravelInsurance"] = df["TravelInsurance"].map({"Not Purchased":0, "Purchased":1})''', language='python')
        df["Employment Type"] = df["Employment Type"].map({"Government Sector": 0, "Private Sector/Self Employed": 1})
        df["GraduateOrNot"] = df["GraduateOrNot"].map({"No": 0, "Yes": 1})
        df["FrequentFlyer"] = df["FrequentFlyer"].map({"No": 0, "Yes": 1})
        df["EverTravelledAbroad"] = df["EverTravelledAbroad"].map({"No": 0, "Yes": 1})
        df["TravelInsurance"] = df["TravelInsurance"].map({"Not Purchased":0, "Purchased":1})
        btn = st.button("Run", key=6)
        if btn:
            st.dataframe(df, hide_index=True)

        st.write('- Checking correlation')
        st.code('''df.corr()''', language='python')
        btn = st.button("Run", key=7)
        if btn:
            st.dataframe(df.corr())

        st.write('')
        st.code('''X = df.iloc[:,:-1]''', language='python')
        X = df.iloc[:,:-1]
        btn = st.button("Run", key=8)
        if btn:
            X

        st.write('')
        st.code('''y = df.iloc[:,-1]''', language='python')
        y = df.iloc[:,-1]
        btn = st.button("Run", key=9)
        if btn:
            y

        st.write('')
        st.code('''scaler = MinMaxScaler() \nXscale = scaler.fit_transform(X) \nXscale''', language='python')
        scaler = MinMaxScaler()
        Xscale = scaler.fit_transform(X)
        btn = st.button("Run", key=10)
        if btn:
            df

        st.write('')
        st.code('''xtrain, xtest, ytrain, ytest = train_test_split(Xscale, y, test_size=0.3, random_state=15)''', language='python')
        Xtrain, Xtest, ytrain, ytest = train_test_split(Xscale, y, test_size=0.3, random_state=15)

        st.write('')
        st.code('''xtrain.shape, xtest.shape''', language='python')
        btn = st.button("Run", key=12)
        if btn:
            Xtrain.shape, Xtest.shape

        st.write('')
        st.code('''ytrain.shape, ytest.shape''', language='python')
        btn = st.button("Run", key=13)
        if btn:
            ytrain.shape, ytest.shape

        # st.write('')
        # st.code('''''', language='python')
        # btn = st.button("Run", key=0)
        # if btn:
        #     st.dataframe(df)


            
        st.code('''model.fit(Xtrain, ytrain)
predictions = model.predict(Xtest)
classification_report(ytest,predictions)''', language='python')
        st.dataframe(pd.read_csv('model_report.csv'),hide_index=True)

    
    with tab02:

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

