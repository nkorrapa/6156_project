# implement nn here
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

data = pd.read_csv('data/T_ONTIME_REPORTING.csv')

data['FL_DATE'] = pd.to_datetime(data['FL_DATE'])
data['DAY_OF_WEEK'] = data['FL_DATE'].dt.day_name()
data.drop(columns=['FL_DATE'], inplace = True)

# want to keep airline_id, origin, dest, carrier dealy, weather dealy, nas delay, security delay, late aircraft delay, crs dep time, crs arrival time, day of week
data_trim = data [['OP_CARRIER_AIRLINE_ID', 'ORIGIN', 'DEST', 'DAY_OF_WEEK', 'CRS_DEP_TIME', 'CRS_ARR_TIME', 'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']]

data_trim = data_trim.fillna(0)
data_trim['TOTAL_DELAY'] = data_trim[['CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']].sum(axis=1)
data_trim['IS_DELAY'] = np.where(data_trim['TOTAL_DELAY'] > 0, 1, 0)
dataset = data_trim[['OP_CARRIER_AIRLINE_ID', 'ORIGIN', 'DEST', 'DAY_OF_WEEK', 'CRS_DEP_TIME', 'CRS_ARR_TIME','IS_DELAY', 'TOTAL_DELAY']]
