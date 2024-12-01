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

def add_colon (string):
  return string [:-2] + ":" + string[-2:]


def linreg_preprocess_data(df_flights):
    df_flights=df_flights[["FL_DATE","OP_CARRIER_AIRLINE_ID","ORIGIN","DEST","CRS_DEP_TIME",
               "DEP_DELAY_NEW","CRS_ARR_TIME","CARRIER_DELAY","WEATHER_DELAY",
               "NAS_DELAY","SECURITY_DELAY", "LATE_AIRCRAFT_DELAY","FIRST_DEP_TIME","TOTAL_ADD_GTIME"]]
    #drop rows where there is not a dept time or arr time
    df_flights['CRS_DEP_TIME'].dropna(inplace=True)
    df_flights['CRS_ARR_TIME'].dropna(inplace=True)
    df_flights['CARRIER_DELAY'].fillna(0, inplace=True)
    df_flights['WEATHER_DELAY'].fillna(0, inplace=True)
    df_flights['NAS_DELAY'].fillna(0, inplace=True)
    df_flights['SECURITY_DELAY'].fillna(0, inplace=True)
    df_flights['LATE_AIRCRAFT_DELAY'].fillna(0, inplace=True)
    df_flights['DEP_DELAY_NEW'].fillna(0, inplace=True)
    #put into time format
    df_flights['CRS_DEP_TIME'] = df_flights['CRS_DEP_TIME'].astype(str).apply(lambda x: x.zfill(4))

    df_flights['CRS_ARR_TIME'] = df_flights['CRS_ARR_TIME'].astype(str).apply(lambda x: x.zfill(4))

    df_flights['CRS_DEP_TIME'] = df_flights['CRS_DEP_TIME'].apply(add_colon)

    df_flights['CRS_ARR_TIME'] = df_flights['CRS_ARR_TIME'].apply(add_colon)

    df_flights['FL_DATE'] = pd.to_datetime(df_flights['FL_DATE'])
    df_flights['day_of_week'] = df_flights['FL_DATE'].dt.day_name()

    #remove date col
    df_flights.drop(columns=['FL_DATE'], inplace = True)

    df_flights['CRS_DEP_TIME_HR'] = df_flights['CRS_DEP_TIME'].str.slice(0, 2)
    df_flights['CRS_ARR_TIME_HR'] = df_flights['CRS_ARR_TIME'].str.slice(0, 2)
    df_flights['TOTAL_DELAY'] = df_flights['DEP_DELAY_NEW'] + df_flights['CARRIER_DELAY'] + df_flights['NAS_DELAY'] + df_flights['WEATHER_DELAY'] + df_flights['SECURITY_DELAY'] + df_flights['LATE_AIRCRAFT_DELAY']
    df_flights['CARRIER_DELAY_BI'] = df_flights['CARRIER_DELAY'].apply(lambda x: 0 if x == 0 else 1)

    df_flights['WEATHER_DELAY_BI'] = df_flights['WEATHER_DELAY'].apply(lambda x: 0 if x == 0 else 1)

    df_flights['NAS_DELAY_BI'] = df_flights['NAS_DELAY'].apply(lambda x: 0 if x == 0 else 1)

    df_flights['SECURITY_DELAY_BI'] = df_flights['SECURITY_DELAY'].apply(lambda x: 0 if x == 0 else 1)

    df_flights['LATE_AIRCRAFT_DELAY_BI'] = df_flights['LATE_AIRCRAFT_DELAY'].apply(lambda x: 0 if x == 0 else 1)
    df_flights['TOTAL_DELAY_BI'] = df_flights['TOTAL_DELAY'].apply(lambda x: 0 if x == 0 else 1)
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
    num_name = ['CARRIER_DELAY_BI','WEATHER_DELAY_BI','NAS_DELAY_BI','SECURITY_DELAY_BI','LATE_AIRCRAFT_DELAY_BI']

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
    return df_linmodel

def train_linreg(data, y):
   x_train, x_val, y_train, y_val = train_test_split(data, y ,test_size=0.3,random_state=0)
   model = LinearRegression().fit(x_train, y_train)
   return model