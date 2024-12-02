# import stuff
import streamlit as st
import numpy as np
import pandas as pd
import time

import datetime
from functions.knn import preprocess_data, train_models
from functions.linreg import add_colon, linreg_preprocess_data, train_linreg
#####################################
@st.cache_data()
def load_file(file):
    time.sleep(3)
    if file is not None:
        df = pd.read_csv(file)
    else:
        df = pd.read_csv(file)
    return(df)
#######################################
# load flight schedule
# load model
# load look up table
# input box
delay = "MODEL OUTPUT"
delay_reason = "KNN model output"
flight_schedule = load_file('data/flight_schedule.csv')
airport_cities = load_file('data/airport_cities.csv')
airport_list = airport_cities[["AIRPORT"]].sort_values("AIRPORT")
airlines = load_file('data/airline_list.csv')
airline_list = airlines[['Description']].sort_values("Description")
today = datetime.datetime.now()
now = datetime.datetime.now()
############################################
st.write("Enter Flight Information")
d = st.date_input("When is your flight", value = now)
t = st.time_input("What time is your flight", value = today)
origin = st.selectbox("Origin Airport", airport_list, index = None, placeholder = "Select Origin airport")
destination = st.selectbox("Destination Airport", airport_list, index = None, placeholder= "Select Destination Airport")
airline = st.selectbox("Airline", airline_list,index = 0, placeholder= "Select Airline")

day_of_week = d.strftime('%A')
airline_id = airlines[airlines['Description'] == airline]['OP_CARRIER_AIRLINE_ID'].item()

input_array = np.array([day_of_week, t, origin, destination, airline_id]) # model input

#############################
#city_origin = airport_cities[airport_cities['AIRPORT'] == origin]['CITY_MARKET'].item()
#city_dest = airport_cities[airport_cities['AIRPORT'] == destination]['CITY_MARKET'].item()

## get similar flights
#filter = np.where((flight_schedule['day_of_week'] == day_of_week) & (flight_schedule['CRS_DEP_TIME'] == t) & (flight_schedule['ORIGIN_CITY_MARKET_ID'] == city_origin) & (flight_schedule['DEST_CITY_MARKET_ID'] == city_dest))
#similar_flights = flight_schedule.loc[filter]


###########################
st.write("Your flight is usually delayed by " + delay_reason + ".")
st.write("Your flight is delayed " + delay + " minutes on average.")

st.write("Other Flight Options:")
st.write("PLACEHOLDER FOR OPTIONS LOGIC")
# avg delay
# other options