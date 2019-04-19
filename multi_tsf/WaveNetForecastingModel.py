import keras
import numpy as np
import os
import shutil
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from multi_tsf.time_series_utils import SyntheticSinusoids, ForecastTimeSeries
from typing import List
import pandas as pd

class WaveNetForecastingModel(object):

    def __init__(self,
                 name: str,
                 nb_layers: int,
                 nb_filters: int,
                 nb_dilation_factors: List[int],
                 nb_input_features: int,
                 nb_output_features: int) -> None:

        self.name = name
        self.nb_layers = nb_layers
        self.nb_filters = nb_filters
        self.nb_dilation_factors = nb_dilation_factors
        self.nb_input_features = nb_input_features
        self.nb_output_features = nb_output_features
        self.placeholder_X = None
        self.placeholder_y = None
        self.dataset = None
        self.iterator = None
        self.data_X = None
        self.data_y = None
        self.pred_y = None
        self.batch_size = None
        self.loss = None
        self.saver = None
        self.optimizer = None
        self.train_writer = None
        self.test_writer = None
        self.model_path = None
        self.meta_path = None


    def _create_model(self, data_X: tf.Tensor, data_y: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        carry = data_X
        skip_connection = tf.keras.layers.Conv1D(filters=self.nb_filters,
                                                 kernel_size=1,
                                                 padding='same',
                                                 activation=tf.keras.activations.relu)
        #Skip connection
        dcc_layer1 = tf.keras.layers.Conv1D(filters=self.nb_filters,
                                               kernel_size=2,
                                               strides=1,
                                               padding='causal',
                                               dilation_rate=1,
                                               activation=keras.activations.relu,
                                               kernel_regularizer=tf.keras.regularizers.l2(l=0.01))
        carry = tf.keras.layers.add([skip_connection(carry), dcc_layer1(carry)])
        for i in range(self.nb_layers):
            dcc_layer = tf.keras.layers.Conv1D(filters=self.nb_filters,
                                               kernel_size=2,
                                               strides=1,
                                               padding='causal',
                                               dilation_rate=self.nb_dilation_factors[i],
                                               activation=keras.activations.relu,
                                               kernel_regularizer=tf.keras.regularizers.l2(l=0.01))
            #Residual Connections
            carry = tf.keras.layers.add([dcc_layer(carry), carry])

        final_dcc_layer = tf.keras.layers.Conv1D(filters=self.nb_output_features,
                                                 kernel_size=1,
                                                 strides=1,
                                                 padding='same',
                                                 activation=keras.activations.relu,
                                                 kernel_regularizer=tf.keras.regularizers.l2(l=0.01))
        self.pred_y = final_dcc_layer(carry)
        loss = tf.math.reduce_mean(tf.keras.losses.mean_absolute_error(self.pred_y, data_y))
        naive_loss = tf.math.reduce_mean(tf.keras.losses.mean_absolute_error(data_X, data_y))
        with tf.name_scope('Loss'):
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('naive_loss', naive_loss)

        return loss, self.pred_y


    def fit(self,
            forecast_data: ForecastTimeSeries,
            model_path: str,
            epochs: int,
            batch_size: int,
            lr: float = 1e-3):
        os.makedirs(model_path, exist_ok=True)
        self.batch_size = batch_size
        self.placeholder_X = tf.placeholder(tf.float32, [None, None, self.nb_input_features])
        self.placeholder_y = tf.placeholder(tf.float32, [None, None, self.nb_output_features])
        self.dataset = tf.data.Dataset.from_tensor_slices((self.placeholder_X, self.placeholder_y))
        self.batched_dataset = self.dataset.batch(batch_size=self.batch_size)

        self.iterator = self.batched_dataset.make_initializable_iterator()
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
                        train_i += 1
                        print(loss)
                    except tf.errors.OutOfRangeError:
                        break

                sess.run(self.iterator.initializer,
                         feed_dict={self.placeholder_X: forecast_data.val_X,
                                    self.placeholder_y: forecast_data.val_y})

                while (True):
                    try:
                        loss, pred_y, data_y, summary = sess.run([self.loss, self.pred_y, self.data_y, merged])
                        self.test_writer.add_summary(summary, val_i)
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
                history: np.array,
                future: np.array,
                nb_steps_out: int) -> np.array:
        proxy_y = np.zeros((history.shape[0], history.shape[1], self.nb_output_features))
        with tf.Session() as sess:
            new_saver = tf.train.import_meta_graph(self.meta_path)
            new_saver.restore(sess, tf.train.latest_checkpoint(self.model_path))
            carry = history
            step_predictions = []
            for i in range(nb_steps_out):
                sess.run(self.iterator.initializer, feed_dict={self.placeholder_X: carry, self.placeholder_y: proxy_y})
                carries = []
                while (True):
                    try:
                        carry = sess.run([self.pred_y])[0]
                        carries.append(carry)
                    except tf.errors.OutOfRangeError:
                        break
                carry = np.vstack(carries)
                step_predictions.append(carry[:, -1, :])

        predictions = np.hstack(step_predictions).flatten()
        future = future.flatten()
        index = np.arange(0, predictions.shape[0])
        sns.lineplot(index[0:336], predictions[0:336], label='prediction')
        sns.lineplot(index[0:336], future[0:336], label='future')
        print(np.mean(np.abs(predictions-future)))
        plt.legend()
        plt.show()
        return predictions


def main():
    epochs = 5
    num_sinusoids = 1
    train_size = 0.7
    val_size = 0.15
    nb_dilation_factors = [1, 2, 4, 8]
    nb_layers = len(nb_dilation_factors)
    nb_filters = 64
    nb_steps_in = 100
    nb_steps_out = None
    target_index = 0

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
                                      nb_input_features=forecast_data.nb_input_features,
                                      nb_output_features=forecast_data.nb_output_features)

    wavenet.fit(forecast_data=forecast_data,
                model_path='./wavenet_test',
                epochs=epochs,
                batch_size=64)

    step_predictions = wavenet.predict(history=forecast_data.time_series, nb_steps_out=24)
    print(step_predictions.shape)

if __name__ == '__main__':
    main()