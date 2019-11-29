import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# train quick explore
train = pd.read_csv('train.csv', nrows=10000)
print(train.shape)
print(train.head())

# test quick explore
test = pd.read_csv('test.csv', nrows = 100)
print(test.shape)

# comparison
print('Train columns:', train.columns)
print('Test columns:', test.columns)

# making some quick charts of the training dataset

train.meter_reading.hist()
plt.show()

# metric definition

def rmsle(y_true, y_pred):
    diffs = np.log(y_true + 1) - np.log(y_pred + 1)
    squares = np.power(diffs, 2)
    err = np.sqrt(np.means(squares))
    return err
