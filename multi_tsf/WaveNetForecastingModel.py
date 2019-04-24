import numpy as np
import os
import shutil
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from multi_tsf.time_series_utils import ForecastTimeSeries, generate_stats
from typing import List


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

            carry = tf.keras.layers.add([leaky_relu(skip_connection(carry)), leaky_relu(dcc_layer1(carry))])
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
                carry = tf.keras.layers.add([leaky_relu(dcc_layer(carry)), carry])

            final_dcc_layer = tf.keras.layers.Conv1D(filters=self.nb_output_features,
                                                     kernel_size=1,
                                                     strides=1,
                                                     padding='same',
                                                     activation=activation,
                                                     kernel_regularizer=regularizer,
                                                     kernel_initializer=initializer,
                                                     kernel_constraint=None)
            self.pred_y = leaky_relu(final_dcc_layer(carry))
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
                carry_i = tf.keras.layers.add([leaky_relu(skip_connection_i(carry_i)), leaky_relu(dcc_layer1_i(carry_i))])
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
                carry = tf.keras.layers.add([leaky_relu(dcc_layer(carry)), carry])

            final_dcc_layer = tf.keras.layers.Conv1D(filters=self.nb_output_features,
                                                     kernel_size=1,
                                                     strides=1,
                                                     padding='same',
                                                     activation=activation,
                                                     kernel_regularizer=regularizer,
                                                     kernel_initializer=initializer)

            self.pred_y = leaky_relu(final_dcc_layer(carry))
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

        for gradient, variable in gradients:
            if gradient is None or variable is None:
                continue
            tf.summary.histogram("gradients/" + variable.name, gradient)
            tf.summary.histogram("variables/" + variable.name, variable)

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
                         feed_dict={self.placeholder_X: forecast_data.reshaped_rolling['Train']['features'],
                                    self.placeholder_y: forecast_data.reshaped_rolling['Train']['targets']})
                while (True):
                    try:
                        _, loss, pred_y, data_y, summary = sess.run([train_op, self.loss, self.pred_y, self.data_y, merged])
                        self.train_writer.add_summary(summary, train_i)
                        train_i += 1
                        print(train_i, loss)
                    except tf.errors.OutOfRangeError:
                        break

                sess.run(self.iterator.initializer,
                         feed_dict={self.placeholder_X: forecast_data.reshaped_rolling['Validation']['features'],
                                    self.placeholder_y: forecast_data.reshaped_rolling['Validation']['targets']})

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
        test_periods = forecast_data.reshaped_periods['Test']
        history = test_periods['features']
        future = test_periods['targets']
        dates = test_periods['dates']
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
        dates = np.hstack(dates)
        predictions = np.concatenate(step_predictions, axis=1)
        self.plot(predictions, future, dates)
        mape, mase, rmse = generate_stats(predictions, future)
        return mape, mase, rmse


    def predict(self,
                time_series: np.array,
                nb_steps_out: int) -> np.array:
        time_series = np.expand_dims(time_series, axis=0)
        proxy_y = np.zeros((time_series.shape[0], time_series.shape[1], self.nb_output_features))
        with tf.Session() as sess:
            new_saver = tf.train.import_meta_graph(self.meta_path)
            new_saver.restore(sess, tf.train.latest_checkpoint(self.model_path))
            carry = time_series
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
        predictions = np.concatenate(step_predictions, axis=1)[0]
        return predictions

    def plot(self, predictions, future, dates):
        nb_steps_out = future.shape[1]
        if self.nb_output_features > 1:
            fig, ax = plt.subplots(self.nb_output_features, 1)
            for i in range(self.nb_output_features):
                predictions_i = predictions[:, :, i].flatten()
                future_i = future[:, :, i].flatten()
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
            sns.lineplot(dates, predictions, label='prediction', ax=ax)
            sns.lineplot(dates, future, label='actual', ax=ax)
            predict_dates = dates[0::nb_steps_out]
            ax.scatter(predict_dates, np.zeros(predict_dates.shape), s=50, c='r')
            ax.grid()
            plt.grid()
            plt.show()

def main():
   pass


if __name__ == '__main__':
    main()