import numpy as np
import pandas as pd

# Load data

def all_data():

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

    X_train = train.loc[:,train.columns != 'meter_reading']
    y_train = train.loc[:,train.columns == 'meter_reading']

    return X_train, y_train

#Eliminar variables
#train.drop('floor_count', inplace=True, axis=1)

def num_data():

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
    train.drop('timestamp', inplace=True, axis = 1)
    train.drop('primary_use', inplace=True, axis = 1)

    X_train = train.loc[:,train.columns != 'meter_reading']
    y_train = train.loc[:,train.columns == 'meter_reading']

    return X_train, y_train
