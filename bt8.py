import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import GradientBoostingRegressor
import pickle



def train_boost8():

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
    train.drop('precip_depth_1_hr', inplace=True, axis = 1)
    train.drop('cloud_coverage', inplace=True, axis = 1)
    train.drop('wind_direction', inplace=True, axis = 1)
    train.drop('dew_temperature', inplace=True, axis = 1)
    train.drop('primary_use', inplace=True, axis = 1)
    train.drop('wind_speed', inplace=True, axis = 1)
    train.drop('building_id', inplace=True, axis = 1)
    train.drop('sea_level_pressure', inplace=True, axis = 1)
    train.drop('wday', inplace=True, axis = 1)

    # Imputation
    train = train.interpolate()

    X_train = train.loc[:,train.columns != 'meter_reading']
    y_train = train.loc[:,train.columns == 'meter_reading']

    return X_train, y_train

def test_boost8():

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
    test.drop('precip_depth_1_hr', inplace=True, axis = 1)
    test.drop('cloud_coverage', inplace=True, axis = 1)
    test.drop('wind_direction', inplace=True, axis = 1)
    test.drop('dew_temperature', inplace=True, axis = 1)
    test.drop('primary_use', inplace=True, axis = 1)
    test.drop('wind_speed', inplace=True, axis = 1)
    test.drop('building_id', inplace=True, axis = 1)
    test.drop('sea_level_pressure', inplace=True, axis = 1)
    test.drop('wday', inplace=True, axis = 1)

    # Imputation
    test = test.interpolate()

    test.set_index('row_id', inplace=True)

    return test



X_train, y_train = train_boost8()
print(X_train.head())
mod_boost8 = GradientBoostingRegressor(max_depth=14)
#mod_boost8.fit(X_train, y_train.values.ravel())

#with open("tree8.pkl", "wb") as f:
#     pickle.dump(mod_boost8, f)


with open("tree4rev.pkl","rb") as f:
    mod_boost4 = pickle.load(f)

print(mod_boost8.feature_importances_)
y_train_pred = mod_boost8.predict(X_train)
print(min(y_train_pred))
y_train_pred = np.maximum(0,y_train_pred)
print(np.sqrt(mean_squared_log_error(y_train,y_train_pred)))


#X_test = test_boost8()
#y_pred = mod_boost8.predict(X_test)
#sub = pd.DataFrame(np.maximum(0,y_pred), index = X_test.index, columns = ['meter_reading'])
#sub.sort_values(by = 'row_id', inplace = True)
#sub.to_csv('./submissionboost8.csv')