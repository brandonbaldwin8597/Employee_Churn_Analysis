import streamlit as st
import pickle
import pandas as pd


st.sidebar.title('Car Price Prediction')
html_temp = """
<div style="background-color:blue;padding:10px">
<h2 style="color:white;text-align:center;">Streamlit ML Cloud App </h2>
</div>"""
st.markdown(html_temp,unsafe_allow_html=True)


age=st.sidebar.selectbox("What is the age of your car:",(0,1,2,3))
hp=st.sidebar.slider("What is the hp_kw of your car?", 40, 300, step=5)
km=st.sidebar.slider("What is the km of your car", 0,350000, step=1000)
gearing_type=st.sidebar.radio('Select gear type',('Automatic','Manual','Semi-automatic'))
car_model=st.sidebar.selectbox("Select model of your car", ('Audi A1', 'Audi A2', 'Audi A3', 'Opel Astra', 'Opel Corsa',
       'Opel Insignia', 'Renault Clio', 'Renault Duster', 'Renault Espace'))


model=pickle.load(open("ridge_model","rb"))



my_dict = {
    "age": age,
    "hp_kW": hp,
    "km": km,
    'Gearing_Type':gearing_type,
    "make_model": car_model
    
}

df = pd.DataFrame.from_dict([my_dict])


st.header("The configuration of your car is below")
st.table(df)

columns= ['age',
 'hp_kW',
 'km',
 'Gearing_Type_Automatic',
 'Gearing_Type_Manual',
 'Gearing_Type_Semi-automatic',
 'make_model_Audi A1',
 'make_model_Audi A2',
 'make_model_Audi A3',
 'make_model_Opel Astra',
 'make_model_Opel Corsa',
 'make_model_Opel Insignia',
 'make_model_Renault Clio',
 'make_model_Renault Duster',
 'make_model_Renault Espace']


df = pd.get_dummies(df).reindex(columns=columns, fill_value=0)

st.subheader("Press predict if configuration is okay")

if st.button("Predict"):
    prediction = model.predict(df)
    st.success("The estimated price of your car is €{}. ".format(int(prediction[0])))
    
