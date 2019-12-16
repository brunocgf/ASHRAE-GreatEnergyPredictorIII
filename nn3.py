import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader


def train_NN():

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
    train.drop('site_id', inplace=True, axis = 1)
    train.drop('sea_level_pressure', inplace=True, axis = 1)

    # Imputation
    train = train.interpolate()

    X_train = train.loc[:,train.columns != 'meter_reading']
    y_train = train.loc[:,train.columns == 'meter_reading']

    return X_train.to_numpy(), y_train.to_numpy()

def test_NN():

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
    test.drop('site_id', inplace=True, axis = 1)
    test.drop('sea_level_pressure', inplace=True, axis = 1)

    # Imputation
    test = test.interpolate()

    test.set_index('row_id', inplace=True)

    return test.to_numpy()




class NN_3(nn.Module):
  def __init__(self, input_size, num_classes):
    super(NN_3, self).__init__()
    self.hidden = nn.Linear(input_size, 100)
    self.relu = nn.ReLU()
    self.linear = nn.Linear(100,num_classes)

  def forward(self, x):
    out = self.hidden(x)
    out = self.relu(out)
    out = self.linear(out)
    return out


X_train, y_train = train_NN()

input_size = 7
num_classes = 1
num_epochs = 5
bs = 100
lr = 0.01

train_data = []

for i in range(20216100):
   train_data.append([X_train[i], y_train[i]])

trainloader = DataLoader(train_data, shuffle=True, batch_size=bs)

model3 = NN_3(input_size, num_classes)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model3.parameters(),lr=lr)


for epoch in range(num_epochs):

  for i, data in enumerate(trainloader,0):
    
    inputs, labels = data

    # Propagacion para adelante
    output = model3(inputs.float())
    loss = loss_function(output, labels.float())
    # Propagcion para atras y paso de optimizacion

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i+1) % 100 == 0:
      print ('Epoca: {}/{}, Paso: {}/{}, Perdida: {:.5f}'.format(epoch+1,num_epochs, i+1, len(trainloader), loss.item()))


with torch.no_grad():
    y_train_pred = model3(Variable(torch.from_numpy(X_train)).float()).data.numpy()


y_train_pred = np.maximum(0,y_train_pred)
print(np.sqrt(mean_squared_log_error(y_train,y_train_pred)))

X_test = test_NN()


with torch.no_grad():
    y_pred = model3(Variable(torch.from_numpy(X_test)).float()).data.numpy()

print(min(y_pred))

sub = pd.DataFrame(np.maximum(0,y_pred), index = X_test.index, columns = ['meter_reading'])
sub.sort_values(by = 'row_id', inplace = True)
sub.to_csv('./submissionNN3.csv')