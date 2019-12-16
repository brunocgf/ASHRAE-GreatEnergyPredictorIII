import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Lasso

def meter_dict():
    md = {0: 'electricity', 1: 'chilledwater', 2: 'steam', 3: 'hotwater'}
    return md

def submit(row,y):
    x = row
    x['meter_reading'] = np.maximum(0,y)
    x.set_index('row_id', inplace = True)
    x.sort_values(inplace = True)
    x.to_csv('./submission.csv')

# Lasso Regression

def train_lasso():

    train = pd.read_csv('./data/train.csv')
    building_metadata = pd.read_csv('./data/building_metadata.csv')
    weather_train = pd.read_csv('./data/weather_train.csv')

    # Sort data for future imputation
    train.sort_values(by=['building_id','timestamp'], inplace=True)

    # Merging data
    train = (train
    .merge(building_metadata, on = 'building_id', how='left')
    .merge(weather_train, on = ['site_id','timestamp'], how='left'))

    #Clear memory
    del building_metadata
    del weather_train

    #Add dates variables
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    train['hour'] = train.timestamp.dt.hour
    train['wday'] = train.timestamp.dt.dayofweek
    train['week'] = train.timestamp.dt.weekofyear

    #Eliminate problematic variables
    train.drop(['timestamp','year_built','floor_count','cloud_coverage','site_id','primary_use','wind_direction','square_feet','dew_temperature','sea_level_pressure','wind_speed','precip_depth_1_hr'], inplace=True, axis = 1)

    # Imputation
    train = train.interpolate()
    train.drop(train[train.hour==0].index, inplace=True)
    train.drop(train[train.hour==1].index, inplace=True)
    train.drop(train[train.hour==2].index, inplace=True)
    train.drop(train[train.hour==3].index, inplace=True)
    train.drop(train[train.hour==4].index, inplace=True)
    train.drop(train[train.hour==5].index, inplace=True)
    train.drop(train[train.hour==7].index, inplace=True)
    train.drop(train[train.hour==8].index, inplace=True)
    train.drop(train[train.hour==9].index, inplace=True)
    train.drop(train[train.hour==10].index, inplace=True)
    train.drop(train[train.hour==12].index, inplace=True)
    train.drop(train[train.hour==13].index, inplace=True)
    train.drop(train[train.hour==14].index, inplace=True)
    train.drop(train[train.hour==15].index, inplace=True)
    train.drop(train[train.hour==17].index, inplace=True)
    train.drop(train[train.hour==18].index, inplace=True)
    train.drop(train[train.hour==19].index, inplace=True)
    train.drop(train[train.hour==20].index, inplace=True)
    train.drop(train[train.hour==21].index, inplace=True)
    train.drop(train[train.hour==23].index, inplace=True)
    train.drop(train[train.week==1].index, inplace=True)
    train.drop(train[train.week==2].index, inplace=True)
    train.drop(train[train.week==3].index, inplace=True)
    train.drop(train[train.week==5].index, inplace=True)
    train.drop(train[train.week==7].index, inplace=True)
    train.drop(train[train.week==9].index, inplace=True)
    train.drop(train[train.week==11].index, inplace=True)
    train.drop(train[train.week==13].index, inplace=True)
    train.drop(train[train.week==15].index, inplace=True)
    train.drop(train[train.week==17].index, inplace=True)
    train.drop(train[train.week==19].index, inplace=True)
    train.drop(train[train.week==21].index, inplace=True)
    train.drop(train[train.week==23].index, inplace=True)
    train.drop(train[train.week==25].index, inplace=True)
    train.drop(train[train.week==27].index, inplace=True)
    train.drop(train[train.week==29].index, inplace=True)
    train.drop(train[train.week==31].index, inplace=True)
    train.drop(train[train.week==33].index, inplace=True)
    train.drop(train[train.week==35].index, inplace=True)
    train.drop(train[train.week==37].index, inplace=True)
    train.drop(train[train.week==39].index, inplace=True)
    train.drop(train[train.week==41].index, inplace=True)
    train.drop(train[train.week==43].index, inplace=True)
    train.drop(train[train.week==45].index, inplace=True)
    train.drop(train[train.week==47].index, inplace=True)
    train.drop(train[train.week==49].index, inplace=True)
    train.drop(train[train.week==51].index, inplace=True)
    train.drop(train[train.week==52].index, inplace=True)
    train.drop(train[train.week==53].index, inplace=True)

    # One Hot Encoding
    encode = OneHotEncoder(drop = 'first')
    catego_var = train.loc[:,['building_id','meter']].to_numpy()
    #catego_var = train.loc[:,['meter']].to_numpy()
    catego_var = encode.fit_transform(catego_var).toarray()
    encode_names = train.building_id.unique().tolist()[1:] + ['meter_1','meter_2','meter_3']
    #encode_names = ['meter_1','meter_2','meter_3']
    encode_var = pd.DataFrame(catego_var, columns = encode_names)

    train.drop('meter', inplace=True, axis = 1)

    train.reset_index(drop=True, inplace=True)
    train = train.join(encode_var)

    # Split train-test
    X_train = train.loc[:,train.columns != 'meter_reading']
    y_train = train.loc[:,train.columns == 'meter_reading']

    return X_train, y_train

def test_lasso():

    test = pd.read_csv('./data/test.csv')
    building_metadata = pd.read_csv('./data/building_metadata.csv')
    weather_test = pd.read_csv('./data/weather_test.csv')

    # Sort data for future imputation
    test.sort_values(by=['building_id','timestamp'], inplace=True)

    # Merging data
    test = (test
    .merge(building_metadata, on = 'building_id', how='left')
    .merge(weather_test, on = ['site_id','timestamp'], how='left'))

    del building_metadata
    del weather_test

    #Add dates variables
    test['timestamp'] = pd.to_datetime(test['timestamp'])
    test['hour'] = test.timestamp.dt.hour
    test['wday'] = test.timestamp.dt.dayofweek
    test['week'] = test.timestamp.dt.weekofyear

    #Eliminate problematic variables
    test.drop(['timestamp','year_built','floor_count','cloud_coverage','site_id','primary_use','wind_direction','square_feet','dew_temperature','sea_level_pressure','wind_speed','precip_depth_1_hr'], inplace=True, axis = 1)

    # Imputation
    test = test.interpolate()
    test.drop(test[test.hour==0].index, inplace=True)
    test.drop(test[test.hour==1].index, inplace=True)
    test.drop(test[test.hour==3].index, inplace=True)
    test.drop(test[test.hour==4].index, inplace=True)
    test.drop(test[test.hour==5].index, inplace=True)
    test.drop(test[test.hour==6].index, inplace=True)
    test.drop(test[test.hour==7].index, inplace=True)
    test.drop(test[test.hour==9].index, inplace=True)
    test.drop(test[test.hour==10].index, inplace=True)
    test.drop(test[test.hour==11].index, inplace=True)
    test.drop(test[test.hour==12].index, inplace=True)
    test.drop(test[test.hour==13].index, inplace=True)
    test.drop(test[test.hour==14].index, inplace=True)
    test.drop(test[test.hour==15].index, inplace=True)
    test.drop(test[test.hour==16].index, inplace=True)
    test.drop(test[test.hour==17].index, inplace=True)
    test.drop(test[test.hour==18].index, inplace=True)
    test.drop(test[test.hour==19].index, inplace=True)
    test.drop(test[test.hour==20].index, inplace=True)
    test.drop(test[test.hour==21].index, inplace=True)
    test.drop(test[test.hour==22].index, inplace=True)
    test.drop(test[test.hour==23].index, inplace=True)

    # One Hot Encoding

    encode = OneHotEncoder(categories='auto',drop = 'first')
    catego_var = test.loc[:,['building_id','meter']].to_numpy()
    catego_var = encode.fit_transform(catego_var).toarray()
    encode_names = test.building_id.unique().tolist()[1:] + ['meter_1','meter_2','meter_3']
    encode_var = pd.DataFrame(catego_var, columns = encode_names)

    test.drop('meter', inplace=True, axis = 1)
    test.reset_index(drop=True,inplace=True)
    test = test.join(encode_var)

    # Add row as set_index
    test.set_index('row_id', inplace=True)

    return test


meter_dict = meter_dict()

X_train, y_train = train_lasso()

mod_lasso = Lasso()
mod_lasso.fit(X_train, y_train)

print(mod_lasso.coef_)
from joblib import dump, load
dump(mod_lasso, 'mod_lasso.joblib') 


#X_test = test_lasso()
#y_pred = mod_lasso.predict(X_test)

#sub = pd.DataFrame(np.maximum(0,y_pred), index = X_test.index, columns = ['meter_reading'])
#sub.sort_values(by = 'row_id', inplace = True)
#sub.to_csv('./submission3.csv')