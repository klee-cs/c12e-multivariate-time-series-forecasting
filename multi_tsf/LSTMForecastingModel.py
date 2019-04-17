import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from multi_tsf.ForecastingModel import ForecastingModel
from multi_tsf.time_series_utils import SyntheticSinusoids, ForecastTimeSeries

class LSTMForecastingModel(ForecastingModel):

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
