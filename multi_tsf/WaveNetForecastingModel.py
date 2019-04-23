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
                 conditional: bool,
                 nb_layers: int,
                 nb_filters: int,
                 nb_dilation_factors: List[int],
                 nb_input_features: int,
                 nb_output_features: int) -> None:

        self.name = name
        self.conditional = conditional
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
        regularizer = tf.keras.regularizers.l2(l=0.01)
        initializer = tf.keras.initializers.he_normal()
        # initializer = tf.keras.initializers.normal(mean=0.0, stddev=0.005)
        activation = tf.keras.activations.linear
        leaky_relu = tf.keras.layers.LeakyReLU()
        #Univariate or Multivariate WaveNet
        if not self.conditional:
            carry = data_X
            # Skip connection
            skip_connection = tf.keras.layers.Conv1D(filters=self.nb_filters,
                                                     kernel_size=1,
                                                     padding='same',
                                                     activation=activation,
                                                     kernel_regularizer=regularizer,
                                                     kernel_initializer=initializer,
                                                     kernel_constraint=None)



            dcc_layer1 = tf.keras.layers.Conv1D(filters=self.nb_filters,
                                                kernel_size=2,
                                                strides=1,
                                                padding='causal',
                                                dilation_rate=1,
                                                activation=activation,
                                                kernel_regularizer=regularizer,
                                                kernel_initializer=initializer,
                                                kernel_constraint=None)

            carry = tf.keras.layers.add([skip_connection(leaky_relu(carry)), dcc_layer1(leaky_relu(carry))])
            for i in range(self.nb_layers):
                dcc_layer = tf.keras.layers.Conv1D(filters=self.nb_filters,
                                                   kernel_size=2,
                                                   strides=1,
                                                   padding='causal',
                                                   dilation_rate=self.nb_dilation_factors[i],
                                                   activation=activation,
                                                   kernel_regularizer=regularizer,
                                                   kernel_initializer=initializer,
                                                   kernel_constraint=None)
                #Residual Connections
                carry = tf.keras.layers.add([dcc_layer(leaky_relu(carry)), leaky_relu(carry)])

            final_dcc_layer = tf.keras.layers.Conv1D(filters=self.nb_output_features,
                                                     kernel_size=1,
                                                     strides=1,
                                                     padding='same',
                                                     activation=activation,
                                                     kernel_regularizer=regularizer,
                                                     kernel_initializer=initializer,
                                                     kernel_constraint=None)
            self.pred_y = final_dcc_layer(leaky_relu(carry))
            if self.nb_output_features > 1:
                loss = tf.math.reduce_mean(tf.math.abs(tf.math.subtract(self.pred_y, data_y)), axis=[0, 1])
                naive_loss = tf.math.reduce_mean(tf.math.abs(tf.math.subtract(data_X, data_y)), axis=[0, 1])
                with tf.name_scope('Loss'):
                    for i in range(self.nb_output_features):
                        print(loss)
                        tf.summary.scalar('loss%d' % i, loss[i])
                        tf.summary.scalar('naive_loss%d' % i, naive_loss[i])
            else:
                loss = tf.math.reduce_mean(tf.keras.losses.mean_absolute_error(self.pred_y, data_y))
                naive_loss = tf.math.reduce_mean(tf.keras.losses.mean_absolute_error(data_X, data_y))
                with tf.name_scope('Loss'):
                    tf.summary.scalar('loss', loss)
                    tf.summary.scalar('naive_loss', naive_loss)

            return loss, self.pred_y

        #Conditional WaveNet
        elif self.conditional:
            carry = data_X
            carries = []
            for i in range(self.nb_input_features):
                carry_i = tf.expand_dims(carry[:, :, i], axis=-1)
                # Skip connection
                skip_connection_i = tf.keras.layers.Conv1D(filters=self.nb_filters,
                                                           kernel_size=1,
                                                           padding='same',
                                                           activation=activation,
                                                           kernel_regularizer=regularizer,
                                                           kernel_initializer=initializer)
                dcc_layer1_i = tf.keras.layers.Conv1D(filters=self.nb_filters,
                                                      kernel_size=2,
                                                      strides=1,
                                                      padding='causal',
                                                      dilation_rate=1,
                                                      activation=activation,
                                                      kernel_regularizer=regularizer,
                                                      kernel_initializer=initializer)
                carry_i = tf.keras.layers.add([skip_connection_i(carry_i), dcc_layer1_i(carry_i)])
                carries.append(carry_i)
            carry = tf.keras.layers.add(carries)
            for i in range(self.nb_layers):
                dcc_layer = tf.keras.layers.Conv1D(filters=self.nb_filters,
                                                   kernel_size=2,
                                                   strides=1,
                                                   padding='causal',
                                                   dilation_rate=self.nb_dilation_factors[i],
                                                   activation=activation,
                                                   kernel_regularizer=regularizer,
                                                   kernel_initializer=initializer)
                # Residual Connections
                carry = tf.keras.layers.add([dcc_layer(carry), carry])

            final_dcc_layer = tf.keras.layers.Conv1D(filters=self.nb_output_features,
                                                     kernel_size=1,
                                                     strides=1,
                                                     padding='same',
                                                     activation=activation,
                                                     kernel_regularizer=regularizer,
                                                     kernel_initializer=initializer)

            self.pred_y = final_dcc_layer(carry)
            if self.nb_output_features > 1:
                loss = tf.math.reduce_mean(tf.math.abs(tf.math.subtract(self.pred_y, data_y)), axis=[0, 1])
                naive_loss = tf.math.reduce_mean(tf.math.abs(tf.math.subtract(data_X, data_y)), axis=[0, 1])
                with tf.name_scope('Loss'):
                    for i in range(self.nb_output_features):
                        print(loss)
                        tf.summary.scalar('loss%d' % i, loss[i])
                        tf.summary.scalar('naive_loss%d' % i, naive_loss[i])
            else:
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
        gradients = self.optimizer.compute_gradients(self.loss)
        # grad_clip_value_min = -10
        # grad_clip_value_max = 10
        # gradients = [
        #     (tf.clip_by_value(grad, clip_value_min=grad_clip_value_min, clip_value_max=grad_clip_value_max), var) \
        #         if grad is not None else (grad, var)
        #     for grad, var in gradients]
        #
        for gradient, variable in gradients:
            if gradient is None or variable is None:
                continue
            tf.summary.histogram("gradients/" + variable.name, gradient)
            tf.summary.histogram("variables/" + variable.name, variable)
        #
        # gradients, variables = zip(*gradients)
        # train_op = self.optimizer.apply_gradients(zip(gradients, variables))
        train_op = self.optimizer.minimize(self.loss)

        self.saver = tf.train.Saver()

        tf.random.set_random_seed(22943)
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

            for _ in range(epochs):
                sess.run(self.iterator.initializer,
                         feed_dict={self.placeholder_X: forecast_data.train_X,
                                    self.placeholder_y: forecast_data.train_y})
                while (True):
                    try:
                        _, loss, pred_y, data_y, summary = sess.run([train_op, self.loss, self.pred_y, self.data_y, merged])
                        self.train_writer.add_summary(summary, train_i)
                        train_i += 1
                        print(train_i, loss)
                    except tf.errors.OutOfRangeError:
                        break

                sess.run(self.iterator.initializer,
                         feed_dict={self.placeholder_X: forecast_data.val_X,
                                    self.placeholder_y: forecast_data.val_y})

                while (True):
                    try:
                        loss, pred_y, data_y, summary = sess.run([self.loss, self.pred_y, self.data_y, merged])
                        self.test_writer.add_summary(summary, val_i)
                        val_i += 1
                    except tf.errors.OutOfRangeError:
                        break

            self.model_path = model_path
            self.saver.save(sess, model_path + '/' + self.name, global_step=epochs)
            self.meta_path = model_path + '/' + self.name + '-%d.meta' % epochs


    def evaluate(self,
                forecast_data: ForecastTimeSeries) -> np.array:
        test_periods = forecast_data.periods['Test']
        history = test_periods['features']
        future = test_periods['targets']
        nb_steps_out = forecast_data.nb_steps_out
        proxy_y = np.zeros((history.shape[0], history.shape[1], self.nb_output_features))
        with tf.Session() as sess:
            new_saver = tf.train.import_meta_graph(self.meta_path)
            new_saver.restore(sess, tf.train.latest_checkpoint(self.model_path))
            carry = history
            step_predictions = []
            for i in range(nb_steps_out):
                sess.run(self.iterator.initializer, feed_dict={self.placeholder_X: carry, self.placeholder_y: proxy_y})
                next_steps = []
                while (True):
                    try:
                        next_step = sess.run([self.pred_y])[0]
                        next_steps.append(next_step)
                    except tf.errors.OutOfRangeError:
                        break
                next_steps = np.expand_dims(np.vstack(next_steps)[:, -1, :], axis=1)
                carry = np.concatenate([carry, next_steps], axis=1)
                carry = carry[:, 1:, :]
                step_predictions.append(next_steps)
        dates = np.hstack(test_periods['predict_dates'])
        predictions = np.concatenate(step_predictions, axis=1)
        self.plot(predictions, future, dates)


    def plot(self, predictions, future, dates=None):
        nb_steps_out = future.shape[1]
        if self.nb_output_features > 1:
            fig, ax = plt.subplots(self.nb_output_features, 1)
            for i in range(self.nb_output_features):
                predictions_i = predictions[:, :, i].flatten()
                future_i = future[:, :, i].flatten()
                if dates is None:
                    index = np.arange(0, predictions_i.shape[0])
                    sns.lineplot(index, predictions_i, label='prediction', ax=ax[i])
                    sns.lineplot(index, future_i, label='actual', ax=ax[i])
                else:
                    sns.lineplot(dates, predictions_i, label='prediction', ax=ax[i])
                    sns.lineplot(dates, future_i, label='actual', ax=ax[i])
                    predict_dates = dates[0::nb_steps_out]
                    ax[i].scatter(predict_dates, np.zeros(predict_dates.shape), s=50, c='r')
                    ax[i].grid()
            plt.legend()
            plt.show()
        else:
            fig, ax = plt.subplots()
            predictions = predictions.flatten()
            future = future.flatten()
            if dates is None:
                index = np.arange(0, predictions.shape[0])
                sns.lineplot(index, predictions, label='prediction', ax=ax)
                sns.lineplot(index, future, label='actual', ax=ax)
            else:
                sns.lineplot(dates, predictions, label='prediction', ax=ax)
                sns.lineplot(dates, future, label='actual', ax=ax)
                predict_dates = dates[0::nb_steps_out]
                ax.scatter(predict_dates, np.zeros(predict_dates.shape), s=50, c='r')
                ax.grid()
            plt.grid()
            plt.show()

        print(np.mean(np.abs(predictions-future)))

def main():
    epochs = 250
    num_sinusoids = 1
    train_size = 0.7
    val_size = 0.15
    conditional = False
    nb_dilation_factors = [1, 2, 4, 8, 16]
    nb_layers = len(nb_dilation_factors)
    nb_filters = 64
    nb_steps_in = 750
    nb_steps_out = 24
    target_index = 0

    # Sinusoid Sample Data
    synthetic_sinusoids = SyntheticSinusoids(num_sinusoids=num_sinusoids,
                                             amplitude=1,
                                             sampling_rate=5000,
                                             length=50000)

    sinusoids = pd.DataFrame(synthetic_sinusoids.sinusoids)

    forecast_data = ForecastTimeSeries(sinusoids,
                                       vector_output_mode=False,
                                       train_size=train_size,
                                       val_size=val_size,
                                       nb_steps_in=nb_steps_in,
                                       nb_steps_out=nb_steps_out,
                                       target_index=target_index,
                                       predict_hour=7,
                                       by_timestamp=False)


    wavenet = WaveNetForecastingModel(name='WaveNet',
                                      conditional=conditional,
                                      nb_layers=nb_layers,
                                      nb_filters=nb_filters,
                                      nb_dilation_factors=nb_dilation_factors,
                                      nb_input_features=forecast_data.nb_input_features,
                                      nb_output_features=forecast_data.nb_output_features)

    wavenet.fit(forecast_data=forecast_data,
                model_path='./wavenet_test',
                epochs=epochs,
                batch_size=64)

    wavenet.evaluate(forecast_data)

if __name__ == '__main__':
    main()