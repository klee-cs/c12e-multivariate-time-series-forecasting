import pandas as pd
import numpy as np
import tensorflow as tf
import math
from datetime import datetime
import keras
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Callable, Optional, List



class ForecastTimeSeries(object):

    def __init__(self, time_series: np.array, dates: Optional[List[datetime]]) -> None:
        self.time_series = time_series
        self.dates = dates

    def create_lags_and_target(self,
                               n_steps_in: int,
                               n_steps_out: int,
                               target_index: Optional[int],
                               train_size: float,
                               val_size: float) -> None:

        train_cutoff = math.ceil(self.time_series.shape[0] * train_size)
        val_cutoff = math.ceil(self.time_series.shape[0] * (train_size + val_size))

        X, y = [], []
        for i in range(self.time_series.shape[0]):
            # find the end of this pattern
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            # check if we are beyond the dataset
            if out_end_ix > len(self.time_series):
                break
            # gather input and output parts of the pattern
            if target_index is not None:
                seq_x, seq_y = self.time_series[i:end_ix, :], self.time_series[end_ix:out_end_ix, target_index]
                seq_y = np.expand_dims(seq_y, axis=1)
            else:
                seq_x, seq_y = self.time_series[i:end_ix, :], self.time_series[end_ix:out_end_ix, :]
            X.append(seq_x)
            y.append(seq_y)

        self.features = np.array(X)
        self.targets = np.array(y)

        self.train_X = self.features[:train_cutoff]
        self.train_y = self.targets[:train_cutoff]

        self.val_X = self.features[train_cutoff:val_cutoff]
        self.val_y = self.targets[train_cutoff:val_cutoff]

        self.test_X = self.features[val_cutoff:]
        self.test_y = self.targets[val_cutoff:]


    def plot_targets(self) -> None:
        pass



class LSTM_Forecasting_Model(object):

    def __init__(self,
                 nb_encoder_layers: int,
                 nb_decoder_layers: int,
                 nb_units: int,
                 nb_steps_in: int,
                 nb_steps_out: int,
                 nb_input_features: int,
                 nb_output_features: int) -> None:
        self.nb_encoder_layers = nb_encoder_layers
        self.nb_decoder_layers = nb_decoder_layers
        self.nb_units = nb_units
        self.nb_steps_in = nb_steps_in
        self.nb_steps_out = nb_steps_out
        self.nb_input_features = nb_input_features
        self.nb_output_features = nb_output_features


    def _create_model(self) -> None:
        encoder_rnn_cells = [keras.layers.LSTMCell(units=self.nb_units) for _ in range(self.nb_encoder_layers)]
        encoder_layer = keras.layers.RNN(encoder_rnn_cells, input_shape=(None, self.nb_steps_in, self.nb_input_features))
        encoder_output = encoder_layer(self.data_X)
        repeat_layer = keras.layers.RepeatVector(self.nb_steps_out)
        repeat_output = repeat_layer(encoder_output)

        decoder_rnn_cells = [keras.layers.LSTMCell(units=self.nb_units) for _ in range(self.nb_decoder_layers)]
        decoder_layer = keras.layers.RNN(decoder_rnn_cells,
                                         return_sequences=True)
        decoder_output = decoder_layer(repeat_output)
        time_distributed = keras.layers.TimeDistributed(keras.layers.Dense(self.nb_output_features, activation=keras.activations.tanh))
        self.pred_y = time_distributed(decoder_output)
        self.loss = tf.losses.mean_squared_error(self.data_y, self.pred_y)


    def fit(self,
            forecast_data: ForecastTimeSeries,
            epochs: int,
            batch_size: int = 64,
            lr: float = 1e-3) -> None:
        self.batch_size = batch_size
        self.placeholder_X = tf.placeholder(tf.float32, [None, self.nb_steps_in, self.nb_input_features])
        self.placeholder_y = tf.placeholder(tf.float32, [None, self.nb_steps_out, self.nb_output_features])

        self.dataset = tf.data.Dataset.from_tensor_slices((self.placeholder_X, self.placeholder_y))
        self.dataset = self.dataset.batch(batch_size=self.batch_size)
        self.iterator = self.dataset.make_initializable_iterator()

        self.data_X, self.data_y = self.iterator.get_next()
        self._create_model()

        self.optimizer = tf.train.AdamOptimizer(lr)
        train_op = self.optimizer.minimize(self.loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for _ in range(epochs):
                sess.run(self.iterator.initializer,
                         feed_dict={self.placeholder_X: forecast_data.train_X, self.placeholder_y: forecast_data.train_y})

                while(True):
                    try:
                        _, loss = sess.run([train_op, self.loss])
                        print(loss)
                    except tf.errors.OutOfRangeError:
                        break




            # Plot Test Set Data
            print('Validation Loss')

            sess.run(self.iterator.initializer,
                     feed_dict={self.placeholder_X: forecast_data.val_X, self.placeholder_y: forecast_data.val_y})
            predicted_ts = []
            actual_ts = []
            while (True):
                try:
                    pred_y, val_y = sess.run([self.pred_y, self.data_y])
                    predicted_ts.append(pred_y)
                    actual_ts.append(val_y)
                    print(np.mean(np.square((pred_y-val_y))))
                except tf.errors.OutOfRangeError:
                    break

            predicted_ts = np.vstack(predicted_ts).reshape(-1, self.nb_output_features)
            actual_ts = np.vstack(actual_ts).reshape(-1, self.nb_output_features)

            fig, axes = plt.subplots(self.nb_output_features, 1)

            for i in range(self.nb_output_features):
                axes[i].plot(predicted_ts[:, i].squeeze(), label='predicted%d' % i)
                axes[i].plot(actual_ts[:, i].squeeze(), label='actual%d' % i)
                axes[i].legend()

            plt.show()


class SyntheticSinusoids(object):

    def __init__(self, num_sinusoids, amplitude, sampling_rate, length):
        self.num_sinusoids = num_sinusoids
        self.amplitude = amplitude
        self.sampling_rate = sampling_rate
        self.length = length
        n = np.arange(0, self.length)

        self.sinusoids = []
        for i in range(num_sinusoids):
            f = np.random.uniform(int(sampling_rate//100), int(self.sampling_rate//30))
            noise_level = np.random.uniform(0.05, 0.1)
            x = self.amplitude*np.cos(2*np.pi*f/self.sampling_rate*n) \
             + np.random.normal(loc=0.0, scale=noise_level*self.amplitude, size=n.shape[0])
            self.sinusoids.append(x.reshape(-1, 1))

        self.sinusoids = np.concatenate(self.sinusoids, axis=1)


    def plot(self):
        for i in range(self.sinusoids.shape[1]):
            plt.plot(self.sinusoids[:, i])
        plt.show()


def main():
    # Load Dataset
    epochs = 10
    batch_size = 64
    nb_steps_in = 100
    nb_steps_out = 1
    target_index = None

    # df = pd.read_csv('effort_by_forecast_skill.csv')
    # df = df.set_index('rec_bus_date')
    # df = df.iloc[-2000:, :]
    # df = df.fillna(value=0)


    #Sinusoid Sample Data
    synthetic_sinusoids = SyntheticSinusoids(num_sinusoids=5,
                                             amplitude=1,
                                             sampling_rate=5000,
                                             length=10000)


    nb_input_features = synthetic_sinusoids.sinusoids.shape[1]

    if target_index == None:
        nb_output_features = nb_input_features
    else:
        nb_output_features = 1

    forecast_data = ForecastTimeSeries(synthetic_sinusoids.sinusoids, dates=None)
    forecast_data.create_lags_and_target(n_steps_in=nb_steps_in,
                                         n_steps_out=nb_steps_out,
                                         target_index=target_index,
                                         train_size=0.7,
                                         val_size=0.15)

    lstm_forecast = LSTM_Forecasting_Model(nb_decoder_layers=2,
                                           nb_encoder_layers=2,
                                           nb_units=100,
                                           nb_steps_in=nb_steps_in,
                                           nb_steps_out=nb_steps_out,
                                           nb_input_features=nb_input_features,
                                           nb_output_features=nb_output_features)

    lstm_forecast.fit(forecast_data=forecast_data,
                      epochs=epochs,
                      batch_size=batch_size)


if __name__ == '__main__':
    main()






