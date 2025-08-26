import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_dataset(df, feature_cols, target_col="Close", time_steps=10):
    X, y = [], []
    data = df[feature_cols].values
    target = df[target_col].values
    for i in range(time_steps, len(df)):
        X.append(data[i-time_steps:i])
        y.append(target[i])
    return np.array(X), np.array(y)

def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model
