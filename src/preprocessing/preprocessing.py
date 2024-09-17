import pandas as pd

def load_and_resample_data(filepath):
    data = pd.read_csv(filepath, index_col=0, parse_dates=[0])
    data_resampled = data.resample('H').sum()
    return data_resampled

def create_time_features(data):
    data['year'] = data.index.year
    data['month'] = data.index.month
    data['day'] = data.index.day
    data['hour'] = data.index.hour
    data['dayofweek'] = data.index.dayofweek
    return data
