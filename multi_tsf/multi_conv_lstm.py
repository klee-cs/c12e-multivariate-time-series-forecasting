import pandas as pd
import numpy as np
from numpy import array
import math
import matplotlib.pyplot as plt
from typing import List, Set, Dict, Tuple, Optional, Iterable, Union
import seaborn as sns
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.utils import Sequence
from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import ConvLSTM2D
from keras.layers import BatchNormalization


class SubsequenceGenerator(Sequence):
    '''
    Sequence Generator Class which generates batches of samples in the shape that a ConvLSTM needs
    '''
    def __init__(self, X: np.array, y: np.array, batch_size: int, n_seq: int, n_steps: int, n_features: int) -> None:
        self.x, self.y = X, y
        self.batch_size = batch_size
        self.n_features = n_features
        self.n_seq = n_seq
        self.n_steps = n_steps

    def __len__(self) -> int:
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx: int) -> np.array:
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
                                               self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
                                               self.batch_size]

        batch_x = batch_x.reshape((batch_x.shape[0], self.n_seq, 1, self.n_steps, self.n_features))
        return batch_x, batch_y


class TimeSeries(object):
    '''
    Basic TimeSeries object which takes care of necessary manipulations depending on model choice
    '''
    def __init__(self, covariates: np.array, target: np.array) -> None:
        self.covariates = covariates
        if self.covariates == None:
            self.univariate = True
        else:
            self.univariate = False
        self.target = target.reshape(-1, 1)

    def lag_features_relative_to_target(self,
                                        covariates: np.array,
                                        target: np.array,
                                        lag: int) -> (np.array, np.array):
        future_target = target[lag:]
        lagged_target = target[:-lag]
        lagged_covariates = covariates[:-lag, :]
        lagged_covariates_and_target = np.concatenate([lagged_covariates, lagged_target], axis=1)
        return lagged_covariates_and_target, future_target



class LSTMTimeSeries(TimeSeries):
    def __init__(self, covariates: np.array, target: np.array) -> None:
        super().__init__(covariates, target)

    def get_rolling_covariate_target_pairs(self, pair_generator: Sequence) -> (np.array, np.array):
        X = np.vstack([X_batch for X_batch, _ in pair_generator])
        y = np.vstack([y_batch for _, y_batch in pair_generator])
        return X, y

    def create_rolling_covariate_target_pairs(self,
                                              lagged_covariates_and_target: np.array,
                                              future_target: np.array,
                                              look_back: int,
                                              batch_size: int) -> TimeseriesGenerator:
        self.batch_sequence_pairs = TimeseriesGenerator(lagged_covariates_and_target,
                                                        future_target,
                                                        length=look_back,
                                                        sampling_rate=1,
                                                        batch_size=batch_size)
        return self.batch_sequence_pairs


    def reshape_rolling_covariate_target_pairs(self,
                                               lagged_covariates_and_target: np.array,
                                               future_target: np.array,
                                               batch_size: int,
                                               n_seq: int,
                                               n_steps: int,
                                               n_features: int) -> SubsequenceGenerator:
        self.batch_subsequence_pairs = SubsequenceGenerator(lagged_covariates_and_target,
                                                            future_target,
                                                            batch_size,
                                                            n_seq,
                                                            n_steps,
                                                            n_features,)
        return self.batch_subsequence_pairs

    def split_train_validation_test(self,
                                    train_size: float,
                                    val_size: float,
                                    look_back: int,
                                    batch_size: int,
                                    n_seq: int,
                                    n_steps: int,
                                    n_features: int,
                                    return_reshaped: bool) -> (Sequence, Sequence, Sequence):
        train_cutoff = math.ceil(self.target.shape[0] * train_size)
        val_cutoff = math.ceil(self.target.shape[0] * (train_size + val_size))


        #Split into Train, Validation, and Test Series
        self.train_target = self.target[:train_cutoff]
        if self.univariate == False:
            self.train_covariates = self.covariates[:train_cutoff]
            self.train_covariates = np.concatenate([self.train_covariates, self.train_target], axis=1)
        else:
            self.train_covariates = self.train_target

        self.val_target = self.target[train_cutoff:val_cutoff]
        if self.univariate == False:
            self.val_covariates = self.covariates[train_cutoff:val_cutoff]
            self.val_covariates = np.concatenate([self.val_covariates, self.val_target], axis=1)
        else:
            self.val_covariates = self.val_target

        self.test_target = self.target[val_cutoff:]
        if self.univariate == False:
            self.test_covariates = self.covariates[val_cutoff:]
            self.test_covariates = np.concatenate([self.test_covariates, self.test_target], axis=1)
        else:
            self.test_covariates = self.test_target

        self.train_generator = self.create_rolling_covariate_target_pairs(self.train_covariates, self.train_target, look_back, batch_size)
        self.val_generator = self.create_rolling_covariate_target_pairs(self.val_covariates, self.val_target, look_back, batch_size)
        self.test_generator = self.create_rolling_covariate_target_pairs(self.test_covariates, self.test_target, look_back, batch_size)

        self.train_X, self.train_y = self.get_rolling_covariate_target_pairs(self.train_generator)
        self.val_X, self.val_y = self.get_rolling_covariate_target_pairs(self.val_generator)
        self.test_X, self.test_y = self.get_rolling_covariate_target_pairs(self.test_generator)

        self.reshaped_train_generator = self.reshape_rolling_covariate_target_pairs(self.train_X, self.train_y, batch_size, n_seq, n_steps, n_features)
        self.reshaped_val_generator = self.reshape_rolling_covariate_target_pairs(self.val_X, self.val_y, batch_size, n_seq, n_steps, n_features)
        self.reshaped_test_generator = self.reshape_rolling_covariate_target_pairs(self.test_X, self.test_y, batch_size, n_seq, n_steps, n_features)

        if return_reshaped:
            return (self.reshaped_train_generator, self.reshaped_val_generator, self.reshaped_test_generator)
        else:
            return (self.train_generator, self.val_generator, self.test_generator)


class WaveNetTimeSeries(TimeSeries):
    def __init__(self, covariates: np.array, target: np.array) -> None:
        super().__init__(covariates, target)


class Multi_ConvLSTM(object):
    def __init__(self):
        pass


if __name__ == '__main__':
    #Load Dataset
    df = pd.read_csv('effort_by_forecast_skill.csv')
    df = df.set_index('rec_bus_date')
    df = df.iloc[-2000:, :]
    df = df.fillna(value = 0)
    covariates = df.loc[:, df.columns != 'AGENT DEATH CLAIMS_RA'].values
    target = df['AGENT DEATH CLAIMS_RA'].values.reshape(-1,1)


    #Parameters
    look_back = 100
    batch_size = 64
    n_seq = 1
    n_steps = 100


    #Create Forecasting Object
    death_claims = LSTMTimeSeries(None, target)
    n_features = 1
    train_generator, val_generator, test_generator = death_claims.split_train_validation_test(train_size=0.7,
                                                                                           val_size=0.15,
                                                                                           look_back=look_back,
                                                                                           batch_size=batch_size,
                                                                                           n_seq=n_seq,
                                                                                           n_steps=n_steps,
                                                                                           n_features=n_features,
                                                                                           return_reshaped=True)

    testing_data = death_claims.get_rolling_covariate_target_pairs(test_generator)
    validation_data = death_claims.get_rolling_covariate_target_pairs(val_generator)
    X_test, y_test = testing_data
    # for i in range(X.shape[0]):
    #     print('pairs')
    #     print(X[i, :, -1])
    #     print(y[i])

    model = Sequential()
    model.add(BatchNormalization(input_shape=(n_seq, 1, n_steps, n_features),
                                 axis=-1,
                                 momentum=0.99,
                                 epsilon=0.001,
                                 center=True,
                                 scale=True,
                                 beta_initializer='zeros',
                                 gamma_initializer='ones',
                                 moving_mean_initializer='zeros',
                                 moving_variance_initializer='ones',
                                 beta_regularizer=None,
                                 gamma_regularizer=None,
                                 beta_constraint=None,
                                 gamma_constraint=None))
    model.add(ConvLSTM2D(filters=64,
                         kernel_size=(1, 2),
                         activation='relu',
                         return_sequences=True,
                         input_shape=(n_seq, 1, n_steps, n_features)))
    model.add(BatchNormalization(axis=-1,
                                 momentum=0.99,
                                 epsilon=0.001,
                                 center=True,
                                 scale=True,
                                 beta_initializer='zeros',
                                 gamma_initializer='ones',
                                 moving_mean_initializer='zeros',
                                 moving_variance_initializer='ones',
                                 beta_regularizer=None,
                                 gamma_regularizer=None,
                                 beta_constraint=None,
                                 gamma_constraint=None))
    model.add(ConvLSTM2D(filters=64,
                         kernel_size=(1, 2),
                         activation='relu',
                         input_shape=(n_seq, 1, n_steps, n_features)))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')



    # fit model
    model.fit_generator(train_generator,
                        epochs=50,
                        verbose=1,
                        validation_data=validation_data,
                        callbacks=[TensorBoard(log_dir='./logs',
                                               histogram_freq=1,
                                               batch_size=batch_size,
                                               write_graph=True,
                                               write_grads=False,
                                               write_images=False,
                                               embeddings_freq=0,
                                               embeddings_layer_names=None,
                                               embeddings_metadata=None,
                                               embeddings_data=None,
                                               update_freq='batch')])

    y_pred = model.predict_generator(test_generator)

    sns.lineplot(np.arange(0, y_pred.shape[0]), y_pred.reshape(-1,), label='predicted')
    sns.lineplot(np.arange(0, y_test.shape[0]), y_test.reshape(-1,), label='actual')
    plt.grid()
    plt.show()



