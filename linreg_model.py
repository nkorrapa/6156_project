import pandas as pd
from datetime import datetime
#check multicollinearity of variables
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing

import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats

pd.set_option('display.max_columns', 50)

df_flights = pd.read_csv("data/T_ONTIME_REPORTING.csv")

df_flights2 = df_flights.copy(deep=True)

df_flights.describe(include='all').T

df_flights.dtypes

#drop column i dont need for lin reg
df_flights = df_flights.drop(columns=['OP_CARRIER_FL_NUM', 'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_CITY_MARKET_ID','DEST_AIRPORT_ID','DEST_AIRPORT_SEQ_ID','DEST_CITY_MARKET_ID'])

df_flights = df_flights.drop(columns=['WHEELS_OFF','WHEELS_ON'])

df_flights = df_flights.drop(columns=['FLIGHTS','DIVERTED'])

df_flights = df_flights.drop(columns=['CRS_ELAPSED_TIME','ACTUAL_ELAPSED_TIME','AIR_TIME'])

#drop rows where there is not a dept time or arr time
df_flights['CRS_DEP_TIME'].dropna(inplace=True)
df_flights['CRS_ARR_TIME'].dropna(inplace=True)
df_flights['CANCELLED'].dropna(inplace=True)

df_flights = df_flights.drop(columns=['CANCELLED','CANCELLATION_CODE'])

df_flights = df_flights.drop(columns=['FIRST_DEP_TIME','TOTAL_ADD_GTIME'])

df_flights = df_flights.drop(columns=['DEP_TIME','DEP_DELAY','TAXI_IN','TAXI_OUT','ARR_TIME','ARR_DELAY'])

df_flights = df_flights.drop(columns=['ARR_DELAY_NEW'])

df_flights['CARRIER_DELAY'].fillna(0, inplace=True)
df_flights['WEATHER_DELAY'].fillna(0, inplace=True)
df_flights['NAS_DELAY'].fillna(0, inplace=True)
df_flights['SECURITY_DELAY'].fillna(0, inplace=True)
df_flights['LATE_AIRCRAFT_DELAY'].fillna(0, inplace=True)
df_flights['DEP_DELAY_NEW'].fillna(0, inplace=True)

#put into time format
df_flights['CRS_DEP_TIME'] = df_flights['CRS_DEP_TIME'].astype(str).apply(lambda x: x.zfill(4))

df_flights['CRS_ARR_TIME'] = df_flights['CRS_ARR_TIME'].astype(str).apply(lambda x: x.zfill(4))

def add_colon (string):
  return string [:-2] + ":" + string[-2:]

df_flights['CRS_DEP_TIME'] = df_flights['CRS_DEP_TIME'].apply(add_colon)

df_flights['CRS_ARR_TIME'] = df_flights['CRS_ARR_TIME'].apply(add_colon)

df_flights.dtypes

df_flights.head(7)

df_flights['FL_DATE'] = pd.to_datetime(df_flights['FL_DATE'])
df_flights['day_of_week'] = df_flights['FL_DATE'].dt.day_name()

#remove date col
df_flights.drop(columns=['FL_DATE'], inplace = True)

df_flights['CRS_DEP_TIME_HR'] = df_flights['CRS_DEP_TIME'].str.slice(0, 2)
df_flights['CRS_ARR_TIME_HR'] = df_flights['CRS_ARR_TIME'].str.slice(0, 2)

df_flights['TOTAL_DELAY'] = df_flights['DEP_DELAY_NEW'] + df_flights['CARRIER_DELAY'] + df_flights['NAS_DELAY'] + df_flights['WEATHER_DELAY'] + df_flights['SECURITY_DELAY'] + df_flights['LATE_AIRCRAFT_DELAY']

df_flights['OP_CARRIER_AIRLINE_ID'] = df_flights['OP_CARRIER_AIRLINE_ID'].replace(19790, 'Delta Airline')
df_flights['OP_CARRIER_AIRLINE_ID'] = df_flights['OP_CARRIER_AIRLINE_ID'].replace(19805, 'American Airline')
df_flights['OP_CARRIER_AIRLINE_ID'] = df_flights['OP_CARRIER_AIRLINE_ID'].replace(20304, 'SkyWest Airline')
df_flights['OP_CARRIER_AIRLINE_ID'] = df_flights['OP_CARRIER_AIRLINE_ID'].replace(20436, 'Frontier Airlines')
df_flights['OP_CARRIER_AIRLINE_ID'] = df_flights['OP_CARRIER_AIRLINE_ID'].replace(20397, 'PSA Airlines')
df_flights['OP_CARRIER_AIRLINE_ID'] = df_flights['OP_CARRIER_AIRLINE_ID'].replace(20416, 'Spirit Air')
df_flights['OP_CARRIER_AIRLINE_ID'] = df_flights['OP_CARRIER_AIRLINE_ID'].replace(20409, 'JetBlue Airways')
df_flights['OP_CARRIER_AIRLINE_ID'] = df_flights['OP_CARRIER_AIRLINE_ID'].replace(19393, 'SouthWest Airlines')
df_flights['OP_CARRIER_AIRLINE_ID'] = df_flights['OP_CARRIER_AIRLINE_ID'].replace(19930, 'Alaska Airlines')
df_flights['OP_CARRIER_AIRLINE_ID'] = df_flights['OP_CARRIER_AIRLINE_ID'].replace(20452, 'Republic Airlines')
df_flights['OP_CARRIER_AIRLINE_ID'] = df_flights['OP_CARRIER_AIRLINE_ID'].replace(20363, 'Endeavor Air')
df_flights['OP_CARRIER_AIRLINE_ID'] = df_flights['OP_CARRIER_AIRLINE_ID'].replace(19690, 'Hawaiian Airlines')

df_flights['CARRIER_DELAY_BI'] = df_flights['CARRIER_DELAY'].apply(lambda x: 0 if x == 0 else 1)

df_flights['WEATHER_DELAY_BI'] = df_flights['WEATHER_DELAY'].apply(lambda x: 0 if x == 0 else 1)

df_flights['NAS_DELAY_BI'] = df_flights['NAS_DELAY'].apply(lambda x: 0 if x == 0 else 1)

df_flights['SECURITY_DELAY_BI'] = df_flights['SECURITY_DELAY'].apply(lambda x: 0 if x == 0 else 1)

df_flights['LATE_AIRCRAFT_DELAY_BI'] = df_flights['LATE_AIRCRAFT_DELAY'].apply(lambda x: 0 if x == 0 else 1)

df_flights.head()

df_flights['TOTAL_DELAY_BI'] = df_flights['TOTAL_DELAY'].apply(lambda x: 0 if x == 0 else 1)

df_flights2['TOTAL_DELAY'] = df_flights2['DEP_DELAY_NEW'] + df_flights2['CARRIER_DELAY'] + df_flights2['NAS_DELAY'] + df_flights2['WEATHER_DELAY'] + df_flights2['SECURITY_DELAY'] + df_flights2['LATE_AIRCRAFT_DELAY']

dummy_data_all = pd.get_dummies(df_flights[['ORIGIN', 'DEST','OP_CARRIER_AIRLINE_ID','day_of_week','CRS_DEP_TIME_HR','CRS_ARR_TIME_HR']],drop_first=True)
dummy_data_all.replace({False:0, True:1},inplace = True)

dummy_data_origin = pd.get_dummies(df_flights[['ORIGIN']],drop_first=True)
dummy_data_origin.replace({False:0, True:1},inplace = True)

dummy_data_dest = pd.get_dummies(df_flights[['DEST']],drop_first=True)
dummy_data_dest.replace({False:0, True:1}, inplace = True)

dummy_data_airline = pd.get_dummies(df_flights[['OP_CARRIER_AIRLINE_ID']],drop_first=True)
dummy_data_airline.replace({False:0, True:1}, inplace = True)

dummy_data_day = pd.get_dummies(df_flights[['day_of_week']],drop_first=True)
dummy_data_day.replace({False:0, True:1}, inplace = True)

dummy_data_dephr= pd.get_dummies(df_flights[['CRS_DEP_TIME_HR']],drop_first=True)
dummy_data_dephr.replace({False:0, True:1}, inplace = True)

dummy_data_arrhr= pd.get_dummies(df_flights[['CRS_ARR_TIME_HR']],drop_first=True)
dummy_data_arrhr.replace({False:0, True:1}, inplace = True)

#cols joining from main set
num_name = ['DISTANCE','CARRIER_DELAY_BI','WEATHER_DELAY_BI','NAS_DELAY_BI','SECURITY_DELAY_BI','LATE_AIRCRAFT_DELAY_BI']

#joing dummy sets with og set = model_data

df_origin_og = pd.concat([df_flights[num_name], dummy_data_origin],axis=1)
df_origin_og.describe(include='all')

#joing dummy sets with og set = model_data

df_withdest = pd.concat([df_origin_og, dummy_data_dest],axis=1)
df_withdest.describe(include='all')

df_withairline = pd.concat([df_withdest, dummy_data_airline],axis=1)
df_withairline.describe(include='all')

df_withday = pd.concat([df_withairline, dummy_data_day],axis=1)
df_withday.describe(include='all')

df_withdephr = pd.concat([df_withday, dummy_data_dephr],axis=1)
df_withdephr.describe(include='all')

df_linmodel = pd.concat([df_withdephr, dummy_data_arrhr],axis=1)

df_linmodel.head(20)



# #took too long to run
# vif_data = pd.DataFrame()
# vif_data["feature"] = df_linmodel.columns
# vif_data["VIF"] = [vif(df_linmodel.values,i) for i in range(len(df_linmodel.columns))]
# print(vif_data)

y = df_flights['TOTAL_DELAY']
y.describe()

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(df_linmodel, y ,test_size=0.3,random_state=0)

model = LinearRegression().fit(x_train, y_train)

predictions = model.predict(x_val)
print(predictions)

print('Score (Training):  {:.3f}'.format(model.score(x_train, y_train)))
print('Score (Test):      {:.3f}'.format(model.score(x_val, y_val)))

print("Coefficients: \n", model.coef_)

lincoeffs = pd.DataFrame(model.coef_, x_train.columns, columns=['Coefficients'])
lincoeffs.head()

x_train_sm = sm.add_constant(x_train)
# Fit the model with statsmodels
est = sm.OLS(y_train, x_train_sm)
est2 = est.fit()
print(est2.summary())

















# df_flights['CRS_DEP_TIME'] = df_flights['CRS_DEP_TIME'].replace('24:00', '23:59')
# df_flights['CRS_ARR_TIME'] = df_flights['CRS_ARR_TIME'].replace('24:00', '23:59')

# df_flights['CRS_DEP_TIME'] = pd.to_datetime(df_flights['CRS_DEP_TIME'], format='%H:%M:%S')

# df_flights['CRS_ARR_TIME'] = pd.to_datetime(df_flights['CRS_ARR_TIME'], format='%H:%M')

# df_flights['CRS_ARR_TIME'] = df_flights['CRS_ARR_TIME'].dt.time

