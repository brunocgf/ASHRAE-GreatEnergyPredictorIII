import pandas as pd

train = pd.read_csv('./data/train.csv')
building_metadata = pd.read_csv('./data/building_metadata.csv')
weather_train = pd.read_csv('./data/weather_train.csv')

    # Meter 0 correction

    # Merging data
train = (train
.merge(building_metadata, on = 'building_id', how='left')
.merge(weather_train, on = ['site_id','timestamp'], how='left'))

print(train.shape())
