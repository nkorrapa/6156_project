# import stuff
import streamlit as st
import numpy as np
import pandas as pd
import time
import datetime

from functions.knn import preprocess_data, train_models

from sklearn.preprocessing import StandardScaler
import keras
#####################################
@st.cache_data()
def load_file(file):
    time.sleep(3)
    if file is not None:
        df = pd.read_csv(file)
    else:
        df = pd.read_csv(file)
    return(df)
def add_colon (string):
  return string [:-2] + ":" + string[-2:]
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
flight_info = load_file('data/flight_info.csv')
############################################
st.write("Enter Flight Information")

d = st.date_input("When is your flight", value = None)
if not d: 
   st.stop()
day_of_week = d.strftime('%A')

origin = st.selectbox("Origin Airport", airport_list, index = None, placeholder = "Select Origin airport")
if not origin:
   st.stop()
   
destination = st.selectbox("Destination Airport", airport_list, index = None, placeholder= "Select Destination Airport")
if not destination:
   st.stop()

airline = st.selectbox("Airline", airline_list, index=None, placeholder= "Select Airline")
if not airline:
  st.stop()
airline_id = airlines[airlines['Description'] == airline]['OP_CARRIER_AIRLINE_ID'].item()

flight_start = flight_schedule[(flight_schedule['day_of_week'] == day_of_week) & (flight_schedule['OP_CARRIER_AIRLINE_ID'] == airline_id) & (flight_schedule['ORIGIN'] == origin) & (flight_schedule['DEST'] == destination)][["CRS_DEP_TIME"]].sort_values("CRS_DEP_TIME")

t = st.selectbox("What time is your flight", flight_start, index = None, placeholder = "Select departure time")
if not t:
   st.stop()


city_origin = airport_cities.loc[airport_cities['AIRPORT'] == origin, 'CITY_MARKET'].item()
city_dest = airport_cities.loc[airport_cities['AIRPORT'] == destination]['CITY_MARKET'].item()

input_array = np.array([day_of_week, t, origin, destination, airline_id]) # model input

nn_model_input = flight_info[(flight_info['DAY_OF_WEEK'] == day_of_week) & (flight_info['ORIGIN'] == origin) & (flight_info['DEST'] == destination) & (flight_info['CRS_DEP_TIME'] == t)][["AIR_TIME","DISTANCE","CRS_ELAPSED_TIME"]]

#############################

## get similar flights
flight_schedule = flight_schedule.merge(airlines, on = "OP_CARRIER_AIRLINE_ID", how = 'left')
flight_schedule['CRS_ARR_TIME']= flight_schedule['CRS_ARR_TIME'].astype(str)
flight_schedule['CRS_ARR_TIME'] = flight_schedule['CRS_ARR_TIME'].apply(add_colon)
flight_schedule['CRS_DEP_TIME']= flight_schedule['CRS_DEP_TIME'].astype(str)
flight_schedule['CRS_DEP_TIME'] = flight_schedule['CRS_DEP_TIME'].apply(add_colon)

similar_flights = flight_schedule[(flight_schedule['day_of_week'] == day_of_week) & (flight_schedule['ORIGIN_CITY_MARKET_ID'] == city_origin) & (flight_schedule['DEST_CITY_MARKET_ID'] == city_dest)][["Description","ORIGIN","DEST","CRS_DEP_TIME","CRS_ARR_TIME"]]


if similar_flights.empty:
    similar_flights = "No other flight options"
else:
    similar_flights.reset_index(drop = True, inplace=True)
    similar_flights.rename(columns = { "Description" : "Airline", "ORIGIN" : "Origin", "DEST" : "Destination", "CRS_DEP_TIME": "Departure Time", "CRS_ARR_TIME": "Arrival Time"}, inplace = True)

##### get NN model output
scaler = StandardScaler()

user_input = nn_model_input.to_numpy()
user_scaled = scaler.transform(user_input)
model = keras.models.load_model('nn_model.keras')
delay = model(user_scaled)

###########################

st.write("Your flight is usually delayed by " + delay_reason + ".")
if delay[0][1]>= 0.5:
  st.write("Your flight is delayed " + delay + " minutes on average.")
  st.write("Other Flight Options:")
  st.write(similar_flights)
else:
   st.write("Your flight is usually on time.")
