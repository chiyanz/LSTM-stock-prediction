import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import TimeDistributed
import tensorflow as tf
import keras
from keras import optimizers
from keras.callbacks import History
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class LSTMPredictor():
  def __init__(self, data, input_labels, output_labels) -> None:
    '''
    Constructs an optimized LSTM network based on the training data, shapes and scales the training data for training, and executes predictions
    '''
    self.raw_data = data
    self.inputs = input_labels
    self.outputs = output_labels
    self.data = self.organize_data(data, input_labels, output_labels)


  def organize_data(self, data, input_labels, output_labels):
    '''
    scales the data and splits the input training data matrix into X (inputs) and Y (outputs) 
    '''
    sc = MinMaxScaler(feature_range=(0,1))
    data = sc.fit_transform(data)
    X, y = data.iloc[input_labels], data.iloc[output_labels]

    # LSTM model takes input in a 3D matrix in (batch_size, timestep, features)
    # TODO: decide on timesteps 
    X = X.reshape(len(X), 1, len(X.columns))

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # train validation split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2
    

    data = {
      'raw': [X, y],
      'train': [X_train, y_train],
      'test': [X_test, X_test],
      'validate': [X_val, y_val]
    }
    return data
  
  
  def optimize_model(self):
    '''
    optimize the hyperparameters of keras LSTM model
    neuron optimization:
      N_h = N_x / (a * (N_i + N_o))
      Nᵢ is the number of input neurons, Nₒ the number of output neurons, Nₛ the number of samples in the training data, and a represents a scaling factor 2-10
    '''
    # using a more simple neuron optimization method to save time
    hidden_nodes = int(2/3 * (len(self.inputs) * len(self.outputs)))
    return hidden_nodes
    

  def compile_model(self, units):
    '''

    params: units: the number of neurons
    '''
    print('Build model...')
    model = Sequential()
    model.add(LSTM(units, return_sequences=True, input_shape=(1, len(self.inputs))))
    model.add(LSTM(units, input_shape=(1, len(self.inputs))))
    # prevent overfitting by adding a dropout layout
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.add(Activation('linear'))

    # mse loss function for optimizing regression-like prediction
    # mean absolute percentage error to evaluate relative magnitude of error
    model.compile(loss='mse', optimizer='adam', metrics=[keras.metrics.MeanAbsolutePercentageError()])

    # batch size = number of inputs to run before using back-propagation to improve weights
    batch_size=16
    model.fit(self.data['train'][0], self.data['train'][1], batch_size=batch_size, epochs=20, validation_data=(self.data['validate'][0], self.data['validate'][1]))


  def driver(self):
    hidden_nodes = self.optimize_model()
    self.compile_model(hidden_nodes)



  