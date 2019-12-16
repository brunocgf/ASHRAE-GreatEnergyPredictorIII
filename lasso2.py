import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Lasso


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
    test.drop(test[test.hour==2].index, inplace=True)
    test.drop(test[test.hour==3].index, inplace=True)
    test.drop(test[test.hour==4].index, inplace=True)
    test.drop(test[test.hour==5].index, inplace=True)
    test.drop(test[test.hour==6].index, inplace=True)
    test.drop(test[test.hour==7].index, inplace=True)
    test.drop(test[test.hour==8].index, inplace=True)
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



#X_train, y_train = train_lasso()

#mod_lasso = Lasso()
#mod_lasso.fit(X_train, y_train)

#print(mod_lasso.coef_)
from joblib import dump, load
mod_lasso = load('mod_lasso.joblib') 


X_test = test_lasso()
y_pred = mod_lasso.predict(X_test)
print(X_test.head())

sub = pd.DataFrame(np.maximum(0,y_pred), index = X_test.index, columns = ['meter_reading'])
sub.sort_values(by = 'row_id', inplace = True)
sub.to_csv('./submission12.csv')