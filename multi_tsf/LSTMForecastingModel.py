import keras
import numpy as np
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
                         nb_steps_in,
                         nb_input_features,
                         nb_output_features)
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
        time_distributed = keras.layers.TimeDistributed(keras.layers.Dense(self.nb_output_features, activation=keras.activations.tanh))
        self.pred_y = time_distributed(decoder_output)
        self.loss = tf.losses.mean_squared_error(self.data_y, self.pred_y)

    def predict(self,
                data_X: np.array) -> np.array:
        data_y = np.zeros((1, self.nb_steps_out, self.nb_output_features))
        with tf.Session() as sess:
            new_saver = tf.train.import_meta_graph(self.meta_path)
            new_saver.restore(sess, tf.train.latest_checkpoint(self.model_path))
            sess.run(self.iterator.initializer,
                     feed_dict={self.placeholder_X: data_X, self.placeholder_y: data_y})
            pred_y = sess.run([self.pred_y])
            return pred_y[0]

    def plot_historical(self,
                        predicted_ts: np.array,
                        actual_ts: np.array) -> None:
        fig1, axes1 = plt.subplots(self.nb_output_features, 2)

        for i in range(self.nb_output_features):
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
    batch_size = 64
    nb_steps_in = 100
    nb_steps_out = 5
    target_index = None

    #Sinusoid Sample Data
    synthetic_sinusoids = SyntheticSinusoids(num_sinusoids=5,
                                             amplitude=1,
                                             sampling_rate=5000,
                                             length=10000)


    forecast_data = ForecastTimeSeries(synthetic_sinusoids.sinusoids)
    forecast_data.create_lags_and_target(n_steps_in=nb_steps_in,
                                         n_steps_out=nb_steps_out,
                                         target_index=target_index)
    forecast_data.split_train_validation_test(train_size=0.7,
                                              val_size=0.15)

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
                                                               plot=True)

    result = lstm_forecast.predict(np.zeros((1, nb_steps_in, 5)))
    print(result[0].shape)


if __name__ == '__main__':
    main()
