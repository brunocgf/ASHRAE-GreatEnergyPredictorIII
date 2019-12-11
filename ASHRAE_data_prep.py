import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder

# First Model

def train_all():

    train = pd.read_csv('./data/train.csv')
    building_metadata = pd.read_csv('./data/building_metadata.csv')
    weather_train = pd.read_csv('./data/weather_train.csv')

    # Meter 0 correction

    # Merging data
    train = (train
    .merge(building_metadata, on = 'building_id', how='left')
    .merge(weather_train, on = ['site_id','timestamp'], how='left'))

    #Add dates variables
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    train['hour'] = train.timestamp.dt.hour
    train['day'] = train.timestamp.dt.normalize()
    train['wday'] = train.timestamp.dt.dayofweek
    train['week'] = train.timestamp.dt.weekofyear

    #Validation data

    X_train = train.loc[:,train.columns != 'meter_reading']
    y_train = train.loc[:,train.columns == 'meter_reading']

    return X_train, y_train

#Eliminar variables
#train.drop('floor_count', inplace=True, axis=1)

def train_reg():

    train = pd.read_csv('./data/train.csv')
    building_metadata = pd.read_csv('./data/building_metadata.csv')
    weather_train = pd.read_csv('./data/weather_train.csv')

    # Sort data for future imputation
    train.sort_values(by=['building_id','timestamp'], inplace=True)

    # Merging data
    train = (train
    .merge(building_metadata, on = 'building_id', how='left')
    .merge(weather_train, on = ['site_id','timestamp'], how='left'))

    #Add dates variables
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    train['hour'] = train.timestamp.dt.hour
    train['wday'] = train.timestamp.dt.dayofweek
    train['week'] = train.timestamp.dt.weekofyear

    #Eliminate problematic variables
    train.drop('timestamp', inplace=True, axis = 1)
    train.drop('primary_use', inplace=True, axis = 1)
    train.drop('year_built', inplace=True, axis = 1)
    train.drop('floor_count', inplace=True, axis = 1)
    train.drop('precip_depth_1_hr', inplace=True, axis = 1)
    train.drop('cloud_coverage', inplace=True, axis = 1)

    # Imputation
    train = train.interpolate()

    X_train = train.loc[:,train.columns != 'meter_reading']
    y_train = train.loc[:,train.columns == 'meter_reading']

    return X_train, y_train


def test_reg():

    test = pd.read_csv('./data/test.csv')
    building_metadata = pd.read_csv('./data/building_metadata.csv')
    weather_test = pd.read_csv('./data/weather_test.csv')

    # Sort data for future imputation
    test.sort_values(by=['building_id','timestamp'], inplace=True)

    # Merging data
    test = (test
    .merge(building_metadata, on = 'building_id', how='left')
    .merge(weather_test, on = ['site_id','timestamp'], how='left'))

    #Add dates variables
    test['timestamp'] = pd.to_datetime(test['timestamp'])
    test['hour'] = test.timestamp.dt.hour
    test['wday'] = test.timestamp.dt.dayofweek
    test['week'] = test.timestamp.dt.weekofyear

    #Eliminate problematic variables
    test.drop('timestamp', inplace=True, axis = 1)
    test.drop('primary_use', inplace=True, axis = 1)
    test.drop('year_built', inplace=True, axis = 1)
    test.drop('floor_count', inplace=True, axis = 1)
    test.drop('precip_depth_1_hr', inplace=True, axis = 1)
    test.drop('cloud_coverage', inplace=True, axis = 1)

    # Imputation
    test = test.interpolate()

    X_test = test.loc[:,test.columns != 'row_id']
    row = test.loc[:,test.columns == 'row_id']

    return row, X_test

# Linear Regression with categories

def train_reg_cat():

    train = pd.read_csv('./data/train.csv')
    building_metadata = pd.read_csv('./data/building_metadata.csv')
    weather_train = pd.read_csv('./data/weather_train.csv')

    # Sort data for future imputation
    train.sort_values(by=['building_id','timestamp'], inplace=True)

    # Merging data
    train = (train
    .merge(building_metadata, on = 'building_id', how='left')
    .merge(weather_train, on = ['site_id','timestamp'], how='left'))

    #Add dates variables
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    train['hour'] = train.timestamp.dt.hour
    train['wday'] = train.timestamp.dt.dayofweek
    train['week'] = train.timestamp.dt.weekofyear

    #Eliminate problematic variables
    train.drop('timestamp', inplace=True, axis = 1)
    train.drop('year_built', inplace=True, axis = 1)
    train.drop('floor_count', inplace=True, axis = 1)
    train.drop('cloud_coverage', inplace=True, axis = 1)
    train.drop('site_id', inplace=True, axis = 1)
    train.drop('primary_use', inplace=True, axis = 1)

    # Imputation
    train = train.interpolate()
    train['precip_depth_1_hr'].fillna(0, inplace=True)

    # One Hot Encoding
    encode = OneHotEncoder(drop = 'first')
    catego_var = train.loc[:,['meter']].to_numpy()
    catego_var = encode.fit_transform(catego_var).toarray()
    encode_names = ['meter_1','meter_2','meter_3']
    encode_var = pd.DataFrame(catego_var, columns = encode_names)

    train.drop('meter', inplace=True, axis = 1)

    train = train.join(encode_var)

    # Split train-test
    X_train = train.loc[:,train.columns != 'meter_reading']
    y_train = train.loc[:,train.columns == 'meter_reading']

    return X_train, y_train


def test_reg_cat():

    test = pd.read_csv('./data/test.csv')
    building_metadata = pd.read_csv('./data/building_metadata.csv')
    weather_test = pd.read_csv('./data/weather_test.csv')

    # Sort data for future imputation
    test.sort_values(by=['building_id','timestamp'], inplace=True)

    # Merging data
    test = (test
    .merge(building_metadata, on = 'building_id', how='left')
    .merge(weather_test, on = ['site_id','timestamp'], how='left'))

    #Add dates variables
    test['timestamp'] = pd.to_datetime(test['timestamp'])
    test['hour'] = test.timestamp.dt.hour
    test['wday'] = test.timestamp.dt.dayofweek
    test['week'] = test.timestamp.dt.weekofyear

    #Eliminate problematic variables
    test.drop('timestamp', inplace=True, axis = 1)
    test.drop('year_built', inplace=True, axis = 1)
    test.drop('floor_count', inplace=True, axis = 1)
    test.drop('cloud_coverage', inplace=True, axis = 1)
    test.drop('site_id', inplace=True, axis = 1)
    test.drop('primary_use', inplace=True, axis = 1)

    # Imputation
    test = test.interpolate()
    test['precip_depth_1_hr'].fillna(0, inplace=True)

    # One Hot Encoding
    encode = OneHotEncoder(drop = 'first')
    catego_var = test.loc[:,['meter']].to_numpy()
    catego_var = encode.fit_transform(catego_var).toarray()
    encode_names = ['meter_1','meter_2','meter_3']
    encode_var = pd.DataFrame(catego_var, columns = encode_names)

    test.drop('meter', inplace=True, axis = 1)

    test = test.join(encode_var)

    # Add row as set_index
    test.set_index('row_id', inplace=True)

    return test


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
