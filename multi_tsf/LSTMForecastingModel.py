import keras
import numpy as np
import os
import shutil
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from multi_tsf.ForecastingModel import ForecastingModel
from multi_tsf.time_series_utils import SyntheticSinusoids, ForecastTimeSeries

class LSTMForecastingModel(object):

    def __init__(self,
                 name: str,
                 nb_units: int,
                 nb_encoder_layers: int,
                 nb_decoder_layers: int,
                 nb_steps_in: int,
                 nb_steps_out: int,
                 nb_input_features: int,
                 nb_output_features: int) -> None:
        super().__init__(name,
                         vector_output_mode=True,
                         nb_steps_in=nb_steps_in,
                         nb_steps_out=nb_steps_out,
                         nb_input_features=nb_input_features,
                         nb_output_features=nb_output_features)
        self.nb_units = nb_units
        self.nb_encoder_layers = nb_encoder_layers
        self.nb_decoder_layers = nb_decoder_layers
        self.nb_steps_out = nb_steps_out

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
        time_distributed = keras.layers.TimeDistributed(keras.layers.Dense(self.nb_output_features, activation=keras.activations.relu))
        self.pred_y = time_distributed(decoder_output)
        self.loss = tf.losses.mean_squared_error(self.data_y, self.pred_y)
        with tf.name_scope('Loss'):
            tf.summary.scalar('loss', self.loss)
        with tf.name_scope('Inputs/Outputs'):
            tf.summary.histogram('input_features', self.data_X)
            tf.summary.histogram('output_targets', self.data_y)


    def fit(self,
            forecast_data: ForecastTimeSeries,
            model_path: str,
            epochs: int,
            batch_size: int = 64,
            lr: float = 1e-3):
        os.makedirs(model_path, exist_ok=True)

        self.batch_size = batch_size

        self.placeholder_X = tf.placeholder(tf.float32, [None, self.nb_steps_in, self.nb_input_features])
        self.placeholder_y = tf.placeholder(tf.float32, [None, self.nb_steps_out, self.nb_output_features])
        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.placeholder_X, self.placeholder_y))
        self.val_dataset = tf.data.Dataset.from_tensor_slices((self.placeholder_X, self.placeholder_y))
        self.train_dataset = self.train_dataset.batch(batch_size=self.batch_size).shuffle(buffer_size=100000)
        self.val_dataset = self.val_dataset.batch(batch_size=self.batch_size)

        self.iterator = self.dataset.make_initializable_iterator()
        self.data_X, self.data_y = self.iterator.get_next()
        self.loss, self.pred_y = self._create_model(self.data_X, self.data_y)

        self.optimizer = tf.train.AdamOptimizer(lr)
        train_op = self.optimizer.minimize(self.loss)

        self.saver = tf.train.Saver()

        with tf.Session() as sess:
            merged = tf.summary.merge_all()
            sess.run(tf.global_variables_initializer())
            try:
                shutil.rmtree(model_path + '/logs/train')
                shutil.rmtree(model_path + '/logs/test')
            except FileNotFoundError as fnf_error:
                pass
            self.train_writer = tf.summary.FileWriter(model_path + '/logs/train', sess.graph)
            self.test_writer = tf.summary.FileWriter(model_path + '/logs/test')

            train_i = 0
            val_i = 0
            plot_pred_y = None
            plot_data_y = None

            for _ in range(epochs):
                sess.run(self.iterator.initializer,
                         feed_dict={self.placeholder_X: forecast_data.train_X,
                                    self.placeholder_y: forecast_data.train_y})
                while (True):
                    try:
                        _, loss, pred_y, data_y, summary = sess.run(
                            [train_op, self.loss, self.pred_y, self.data_y, merged])
                        self.train_writer.add_summary(summary, train_i)
                        print(loss)
                        train_i += 1
                    except tf.errors.OutOfRangeError:
                        break

                sess.run(self.iterator.initializer,
                         feed_dict={self.placeholder_X: forecast_data.val_X,
                                    self.placeholder_y: forecast_data.val_y})

                while (True):
                    try:
                        loss, pred_y, data_y, summary = sess.run([self.loss, self.pred_y, self.data_y, merged])
                        self.test_writer.add_summary(summary, val_i)
                        print(loss)
                        plot_pred_y = pred_y
                        plot_data_y = data_y
                        val_i += 1
                    except tf.errors.OutOfRangeError:
                        break

            index = np.arange(0, plot_pred_y.shape[1])
            sns.lineplot(index, plot_pred_y[0, :, 0].reshape(-1, ), label='predicted')
            sns.lineplot(index, plot_data_y[0, :, 0].reshape(-1, ), label='actual')
            plt.legend()
            plt.title('Validation Set')
            plt.show()

            self.model_path = model_path
            self.saver.save(sess, model_path + '/' + self.name, global_step=epochs)
            self.meta_path = model_path + '/' + self.name + '-%d.meta' % epochs

    def predict(self,
                data_X: np.array) -> np.array:
        data_X = np.expand_dims(data_X, axis=0)
        data_y = np.zeros((1, self.nb_steps_out, self.nb_output_features))
        with tf.Session() as sess:
            new_saver = tf.train.import_meta_graph(self.meta_path)
            new_saver.restore(sess, tf.train.latest_checkpoint(self.model_path))
            sess.run(self.val_test_iterator,
                     feed_dict={self.placeholder_X: data_X, self.placeholder_y: data_y})
            pred_y = sess.run([self.pred_y])
            return pred_y[0]

    def plot_historical(self,
                        predicted_ts: np.array,
                        actual_ts: np.array,
                        num_features_display: int = None) -> None:
        if num_features_display is None:
            num_features_display = self.nb_output_features
        fig1, axes1 = plt.subplots(num_features_display, 2)



        for i in range(num_features_display):
            axes1[i, 0].plot(predicted_ts[:, i].squeeze(), label='predicted%d' % i)
            axes1[i, 0].plot(actual_ts[:, i].squeeze(), label='actual%d' % i)
            axes1[i, 0].legend()

            lag_errors = []
            for j in range(self.nb_steps_out):
                pred_lagj = predicted_ts[j::self.nb_steps_out, i]
                actual_lagj = actual_ts[j::self.nb_steps_out, i]
                error_lagj = np.mean(np.square(pred_lagj - actual_lagj))
                lag_errors.append(error_lagj)
            sns.barplot(x=np.arange(1, self.nb_steps_out + 1), y=lag_errors, ax=axes1[i, 1])
        plt.show()


    def predict_historical(self,
                           forecast_data: ForecastTimeSeries,
                           set: str = 'Validation',
                           plot: bool =False,
                           num_features_display: int = None):
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

            sess.run(self.val_iterator,
                     feed_dict={self.placeholder_X: _X, self.placeholder_y: _y})
            predicted_ts = []
            actual_ts = []
            while(True):
                try:
                    pred_y, val_y = sess.run([self.pred_y, self.data_y])
                    predicted_ts.append(pred_y)
                    actual_ts.append(val_y)
                except tf.errors.OutOfRangeError:
                    break

            predicted_ts = np.vstack(predicted_ts).reshape(-1, self.nb_output_features)
            actual_ts = np.vstack(actual_ts).reshape(-1, self.nb_output_features)

            if plot == True:
                self.plot_historical(predicted_ts, actual_ts, num_features_display)

            return predicted_ts, actual_ts


def main():
    # Load Dataset
    epochs = 1
    train_size = 0.7
    val_size = 0.15
    batch_size = 64
    nb_steps_in = 100
    nb_steps_out = 5
    target_index = None

    #Sinusoid Sample Data
    synthetic_sinusoids = SyntheticSinusoids(num_sinusoids=5,
                                             amplitude=1,
                                             sampling_rate=5000,
                                             length=10000)
    sinusoids = pd.DataFrame(synthetic_sinusoids.sinusoids)


    forecast_data = ForecastTimeSeries(sinusoids,
                                       vector_output_mode=True,
                                       train_size=train_size,
                                       val_size=val_size,
                                       nb_steps_in=nb_steps_in,
                                       nb_steps_out=nb_steps_out,
                                       target_index=target_index)

    lstm_forecast = LSTMForecastingModel(name='LSTM',
                                         nb_decoder_layers=2,
                                         nb_encoder_layers=2,
                                         nb_units=100,
                                         nb_steps_in=nb_steps_in,
                                         nb_steps_out=nb_steps_out,
                                         nb_input_features=forecast_data.nb_input_features,
                                         nb_output_features=forecast_data.nb_output_features)

    lstm_forecast.fit(forecast_data=forecast_data,
                      model_path='./lstm_test',
                      epochs=epochs,
                      batch_size=batch_size)

    predicted_ts, actual_ts = lstm_forecast.predict_historical(forecast_data=forecast_data,
                                                               set='Validation',
                                                               plot=True,
                                                               num_features_display=5)

    result = lstm_forecast.predict(np.zeros((nb_steps_in, 5)))
    print(result[0].shape)


if __name__ == '__main__':
    main()
