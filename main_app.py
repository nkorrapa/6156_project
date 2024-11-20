# import stuff
import streamlit as st
import numpy as np
import pandas as pd
import time
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

flight_schedule = load_file('data/flight_schedule.csv')
st.write("hi, pls put in your flight info")
#############################

## All the Logic and stuff

###########################

# avg delay
# other options