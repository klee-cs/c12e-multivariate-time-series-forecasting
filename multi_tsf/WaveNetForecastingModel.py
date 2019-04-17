import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from multi_tsf.ForecastingModel import ForecastingModel
from multi_tsf.time_series_utils import SyntheticSinusoids, ForecastTimeSeries
from typing import List
import pandas as pd

class WaveNetForecastingModel(ForecastingModel):

    def __init__(self,
                 name: str,
                 nb_layers: int,
                 nb_filters: int,
                 nb_dilation_factors: List[int],
                 max_look_back: int,
                 nb_input_features: int,
                 nb_output_features: int) -> None:

        self.nb_layers = nb_layers
        self.nb_filters = nb_filters
        self.nb_dilation_factors = nb_dilation_factors
        self.nb_steps_out = None
        super().__init__(name=name,
                         vector_output_mode=False,
                         nb_steps_in=max_look_back-1,
                         nb_steps_out=None,
                         nb_input_features=nb_input_features,
                         nb_output_features=nb_output_features)

    #@TODO Enhance architecture
    def _create_model(self) -> None:
        carry = self.data_X
        for i in range(self.nb_layers):
            dcc_layer = tf.keras.layers.Conv1D(filters=self.nb_filters,
                                               kernel_size=2,
                                               strides=1,
                                               padding='causal',
                                               dilation_rate=self.nb_dilation_factors[i],
                                               activation=keras.activations.relu)
            #Residual Connections
            if i > 0:
                carry = tf.keras.layers.add([dcc_layer(carry), carry])
            else:
                carry = dcc_layer(carry)
        time_distributed = keras.layers.TimeDistributed(keras.layers.Dense(self.nb_output_features, activation=keras.activations.relu))
        self.pred_y = time_distributed(carry)
        self.loss = tf.losses.mean_squared_error(self.data_y, self.pred_y)
        with tf.name_scope('Loss'):
            tf.summary.scalar('loss', self.loss)

    def plot_historical(self, predicted_ts: np.array, actual_ts: np.array, num_features_display: int = None) -> None:
        if num_features_display is None:
            num_features_display = self.nb_output_features
        fig1, axes1 = plt.subplots(num_features_display, 1)
        predicted_ts = predicted_ts[0::self.nb_steps_in]
        actual_ts = actual_ts[0::self.nb_steps_in]

        if num_features_display == 1:
            axes1.plot(predicted_ts.squeeze(), label='predicted')
            axes1.plot(actual_ts.squeeze(), label='actual')
            axes1.legend()
        else:
            for i in range(num_features_display):
                axes1[i].plot(predicted_ts[:, i].squeeze(), label='predicted%d' % i)
                axes1[i].plot(actual_ts[:, i].squeeze(), label='actual%d' % i)
                axes1[i].legend()

        plt.show()

    def predict(self, data_X: np.array, nb_steps_out: int) -> np.array:
        data_X = np.expand_dims(data_X, axis=0)
        data_y = np.zeros((1, self.nb_steps_in, self.nb_output_features))
        with tf.Session() as sess:
            new_saver = tf.train.import_meta_graph(self.meta_path)
            new_saver.restore(sess, tf.train.latest_checkpoint(self.model_path))

            carry = data_X
            step_predictions = []
            for i in range(nb_steps_out):
                sess.run(self.val_test_iterator,
                         feed_dict={self.placeholder_X: carry, self.placeholder_y: data_y})
                carry = sess.run([self.pred_y])[0]
                step_predictions.append(carry[:, -1, :])

            step_predictions = np.concatenate(step_predictions, axis=0)

            return step_predictions


def main():
    epochs = 5
    num_sinusoids = 5
    train_size = 0.7
    val_size = 0.15
    batch_size = 64
    nb_dilation_factors = [1, 2, 4, 8]
    nb_layers = len(nb_dilation_factors)
    nb_filters = 64
    nb_steps_in = 100
    nb_steps_out = None
    target_index = None

    # Sinusoid Sample Data
    synthetic_sinusoids = SyntheticSinusoids(num_sinusoids=num_sinusoids,
                                             amplitude=1,
                                             sampling_rate=5000,
                                             length=5000)

    sinusoids = pd.DataFrame(synthetic_sinusoids.sinusoids)

    forecast_data = ForecastTimeSeries(sinusoids,
                                       vector_output_mode=False,
                                       train_size=train_size,
                                       val_size=val_size,
                                       nb_steps_in=nb_steps_in,
                                       nb_steps_out=nb_steps_out,
                                       target_index=target_index)

    wavenet = WaveNetForecastingModel(name='WaveNet',
                                      nb_layers=nb_layers,
                                      nb_filters=nb_filters,
                                      nb_dilation_factors=nb_dilation_factors,
                                      max_look_back=nb_steps_in,
                                      nb_input_features=forecast_data.nb_input_features,
                                      nb_output_features=forecast_data.nb_output_features)

    wavenet.fit(forecast_data=forecast_data,
                          model_path='./wavenet_test',
                          epochs=epochs,
                          batch_size=batch_size)

    predicted_ts, actual_ts = wavenet.predict_historical(forecast_data=forecast_data,
                                                         set='Validation',
                                                         plot=True,
                                                         num_features_display=num_sinusoids)

    test_input = synthetic_sinusoids.sinusoids[-99:, :]
    result = wavenet.predict(test_input, nb_steps_out=5)
    pd.DataFrame(result).plot()
    plt.show()

if __name__ == '__main__':
    main()