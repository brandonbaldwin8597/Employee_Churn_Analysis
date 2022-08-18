import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


st.sidebar.title('Churn Prediction')
html_temp = """
<div style="background-color:blue;padding:10px">
<h2 style="color:white;text-align:center;">Streamlit ML Cloud App </h2>
</div>"""
st.markdown(html_temp,unsafe_allow_html=True)


satisfaction_level = st.sidebar.slider("What is the satisfaction level?", 0.0, 1.0, step=0.01)
number_project = st.sidebar.radio("Number of project", (2, 3, 4, 5, 6, 7))
time_spend_company = st.sidebar.radio("How many years worked in the company?", (2, 3, 4, 5, 6, 7, 8, 10))
last_evaluation = st.sidebar.slider('What is the last evaluation score?', 0.0, 1.0, step=0.01)
st.sidebar.subheader("Salary Levels:  Low(0.0), Medium(1.0), High(2.0)")
salary = st.sidebar.selectbox("What is employee salary level?", [0.0, 1.0, 2.0])
average_montly_hours = st.sidebar.slider("What is the working hours per month?", 100, 310, step=10)
#if st.button("Enter"):
    #st.write("Hello {}".format(average_montly_hours.title()))

model = pickle.load(open("gradient_boosting_model","rb"))



my_dict = {
    "satisfaction_level": satisfaction_level,
    "number_project": number_project,
    "time_spend_company": time_spend_company,
    'last_evaluation':last_evaluation,
    "salary": salary,
    "average_montly_hours": average_montly_hours 
}

df = pd.DataFrame.from_dict([my_dict])


st.header("Which factor effected the who employees left or not?")
st.table(df)

columns = ['satisfaction_level',
          'last_evaluation',
          'number_project',
          'average_montly_hours',
          'salary',
          'time_spend_company']

                                      
# enc = OrdinalEncoder()
# df["salary"] = enc.fit_transform(df["salary"])
# df = pd.get_dummies(df).reindex(columns=columns, fill_value=0)

st.subheader("Press predict if configuration is okay")

if st.button("Predict"):
    prediction = model.predict(df)
    value = int(prediction[0])
    if value == 0:
        value = "STAY"
    else:
        value = "LEAVE"
    st.success("If employees will leave or not: {}.".format(value))
    
