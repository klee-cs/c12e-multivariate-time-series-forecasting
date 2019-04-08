import os
import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
from multi_tsf.time_series_utils import SyntheticSinusoids, ForecastTimeSeries


class Forecasting_Mode(object):
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
            model_path: str,
            epochs: int,
            batch_size: int = 64,
            lr: float = 1e-3) -> None:

        os.makedirs(model_path, exist_ok=True)

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

        self.saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for _ in range(epochs):
                sess.run(self.iterator.initializer,
                         feed_dict={self.placeholder_X: forecast_data.train_X,
                                    self.placeholder_y: forecast_data.train_y})

                train_losses = []
                while(True):
                    try:
                        _, loss = sess.run([train_op, self.loss])
                        train_losses.append(loss)
                    except tf.errors.OutOfRangeError:
                        break

                print('Train MSE')
                train_mse = np.mean(train_losses)
                print(train_mse)

                sess.run(self.iterator.initializer,
                         feed_dict={self.placeholder_X: forecast_data.val_X,
                                    self.placeholder_y: forecast_data.val_y})

                val_losses = []
                while(True):
                    try:
                        loss = sess.run([self.loss])
                        val_losses.append(loss)
                    except tf.errors.OutOfRangeError:
                        break
                val_mse = np.mean(val_losses)
                print('Validaiton MSE')
                print(val_mse)

            self.model_path = model_path
            self.saver.save(sess, model_path + '/LSTM_Forecasting_Model', global_step=epochs)
            self.meta_path = model_path + '/LSTM_Forecasting_Model-%d.meta' % epochs


    def predict_historical(self,
                forecast_data: ForecastTimeSeries,
                set: str = 'Validation',
                plot=False):
        if set == 'Validation':
            _X, _y = forecast_data.val_X, forecast_data.val_y
        elif set == 'Test':
            _X, _y = forecast_data.test_X, forecast_data.test_y
        elif set == 'Train':
            _X, _y = forecast_data.train_X, forecast_data.train_y
        else:
            _X, _y = forecast_data.val_X, forecast_data.val_y
        with tf.Session() as sess:
            new_saver = tf.train.import_meta_graph(self.meta_path)
            new_saver.restore(sess, tf.train.latest_checkpoint(self.model_path))

            sess.run(self.iterator.initializer,
                     feed_dict={self.placeholder_X: _X, self.placeholder_y: _y})
            predicted_ts = []
            actual_ts = []
            while (True):
                try:
                    pred_y, val_y = sess.run([self.pred_y, self.data_y])
                    predicted_ts.append(pred_y)
                    actual_ts.append(val_y)
                except tf.errors.OutOfRangeError:
                    break

            predicted_ts = np.vstack(predicted_ts).reshape(-1, self.nb_output_features)
            actual_ts = np.vstack(actual_ts).reshape(-1, self.nb_output_features)
            print('MSE')
            print(np.mean(np.square(predicted_ts - actual_ts)))

            if plot == True:
                fig1, axes1 = plt.subplots(self.nb_output_features, 2, sharex='col')



                for i in range(self.nb_output_features):

                    axes1[i, 0].plot(predicted_ts[:, i].squeeze(), label='predicted%d' % i)
                    axes1[i, 0].plot(actual_ts[:, i].squeeze(), label='actual%d' % i)
                    axes1[i, 0].legend()

                    lag_errors = []
                    for j in range(self.nb_steps_out):
                        pred_lagj = predicted_ts[j::self.nb_steps_out, i]
                        actual_lagj = actual_ts[j::self.nb_steps_out, i]
                        error_lagj = np.mean(np.square(pred_lagj-actual_lagj))
                        lag_errors.append(error_lagj)
                    sns.barplot(x=np.arange(1, self.nb_steps_out-1), y=lag_errors, ax=axes1[i, 1])
                plt.show()




            return predicted_ts, actual_ts

    def predict(self,
                data_X: np.array) -> np.array:
        data_y = np.zeros((1, self.nb_steps_out, self.nb_output_features))
        with tf.Session() as sess:
            new_saver = tf.train.import_meta_graph(self.meta_path)
            new_saver.restore(sess, tf.train.latest_checkpoint(self.model_path))
            sess.run(self.iterator.initializer,
                     feed_dict={self.placeholder_X: data_X, self.placeholder_y: data_y})
            pred_y = sess.run([self.pred_y])
            return pred_y


def main():
    # Load Dataset
    epochs = 1
    batch_size = 64
    nb_steps_in = 100
    nb_steps_out = 5
    target_index = None

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

    forecast_data = ForecastTimeSeries(synthetic_sinusoids.sinusoids)
    forecast_data.create_lags_and_target(n_steps_in=nb_steps_in,
                                         n_steps_out=nb_steps_out,
                                         target_index=target_index)
    forecast_data.split_train_validation_test(train_size=0.7,
                                              val_size=0.15)

    lstm_forecast = LSTM_Forecasting_Model(nb_decoder_layers=2,
                                           nb_encoder_layers=2,
                                           nb_units=100,
                                           nb_steps_in=nb_steps_in,
                                           nb_steps_out=nb_steps_out,
                                           nb_input_features=nb_input_features,
                                           nb_output_features=nb_output_features)

    lstm_forecast.fit(forecast_data=forecast_data,
                      model_path='./lstm_test',
                      epochs=epochs,
                      batch_size=batch_size)

    predicted_ts, actual_ts = lstm_forecast.predict_historical(forecast_data=forecast_data,
                                                                 set='Validation',
                                                                 plot=True)


if __name__ == '__main__':
    main()
