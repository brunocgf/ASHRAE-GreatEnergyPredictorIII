import numpy as np

def meter_dict():
    md = {0: 'electricity', 1: 'chilledwater', 2: 'steam', 3: 'hotwater'}
    return md

def submit(row,y):
    x = row
    x['meter_reading'] = np.maximum(0,y)
    x.set_index('row_id', inplace = True)
    x.sort_values(inplace = True)
    x.to_csv('./submission.csv')
