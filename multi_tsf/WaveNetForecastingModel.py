import numpy as np
import os
import shutil
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from multi_tsf.time_series_utils import ForecastTimeSeries, generate_stats
from typing import List
from pprint import pprint
from tqdm import tqdm

BATCH_INDEX = 0
TIME_INDEX = 1
CHANNEL_INDEX = 2

class WaveNetForecastingModel(object):

    def __init__(self,
                 name: str,
                 conditional: bool,
                 nb_layers: int,
                 nb_filters: int,
                 nb_dilation_factors: List[int],
                 nb_input_features: int,
                 batch_size: int,
                 lr: float,
                 model_path: str,
                 ts_names: dict) -> None:

        self.ts_names = [x.replace(' ', '-') for x in ts_names]
        self.model_json = {}
        self.model_path = model_path
        self.name = name
        self.conditional = conditional
        self.nb_layers = nb_layers
        self.nb_filters = nb_filters
        self.nb_dilation_factors = nb_dilation_factors
        if conditional:
            self.nb_input_features = nb_input_features
        else:
            self.nb_input_features = 1
        os.makedirs(model_path, exist_ok=True)
        self.batch_size = batch_size
        self.placeholder_X = tf.placeholder(tf.float32, [None, None, self.nb_input_features])
        self.placeholder_y = tf.placeholder(tf.float32, [None, None, 1])
        self.dataset = tf.data.Dataset.from_tensor_slices((self.placeholder_X, self.placeholder_y))
        self.batched_dataset = self.dataset.batch(batch_size=self.batch_size)

        self.iterator = self.batched_dataset.make_initializable_iterator()
        self.data_X, self.data_y = self.iterator.get_next()

        for idx, name in enumerate(self.ts_names):
            with tf.name_scope(name):
                with tf.variable_scope(name):
                    loss, pred_y, summaries = self._create_model(self.data_X,
                                                                  self.data_y,
                                                                  conditional=self.conditional,
                                                                  nb_filters=self.nb_filters,
                                                                  nb_layers=self.nb_layers,
                                                                  nb_dilation_factors=self.nb_dilation_factors,
                                                                  nb_input_features=self.nb_input_features)
                    optimizer = tf.train.AdamOptimizer(lr)
                    gradients = optimizer.compute_gradients(loss)

                    for gradient, variable in gradients:
                        if gradient is None or variable is None:
                            continue
                        summaries.append(tf.summary.histogram("gradients/" + variable.name, gradient))
                        summaries.append(tf.summary.histogram("variables/" + variable.name, variable))

                    train_op = optimizer.minimize(loss)


                    graph_elements = {
                        'index': idx,
                        'loss': loss,
                        'pred_y': pred_y,
                        'optimizer': optimizer,
                        'train_op': train_op,
                        'summaries': summaries
                    }

                    self.model_json[name] = graph_elements


    def _create_model(self,
                      data_X: tf.Tensor,
                      data_y: tf.Tensor,
                      conditional: bool,
                      nb_filters: int,
                      nb_layers: int,
                      nb_dilation_factors: List[int],
                      nb_input_features: int) -> (tf.Tensor, tf.Tensor):

        regularizer = tf.keras.regularizers.l2(l=0.01)
        initializer = tf.keras.initializers.he_normal()
        activation = tf.keras.activations.linear
        leaky_relu = tf.keras.layers.LeakyReLU()

        #Univariate or Multivariate WaveNet
        if not conditional:
            carry = data_X
            # Skip connection
            skip_connection = tf.keras.layers.Conv1D(filters=nb_filters,
                                                     kernel_size=1,
                                                     padding='same',
                                                     activation=activation,
                                                     kernel_regularizer=regularizer,
                                                     kernel_initializer=initializer,
                                                     kernel_constraint=None)



            dcc_layer1 = tf.keras.layers.Conv1D(filters=nb_filters,
                                                kernel_size=2,
                                                strides=1,
                                                padding='causal',
                                                dilation_rate=1,
                                                activation=activation,
                                                kernel_regularizer=regularizer,
                                                kernel_initializer=initializer,
                                                kernel_constraint=None)
            with tf.name_scope('Initial-Layer'):
                carry = tf.keras.layers.add([leaky_relu(skip_connection(carry)), leaky_relu(dcc_layer1(carry))])
            for i in range(nb_layers):
                with tf.name_scope('Dilated-Stack'):
                    dcc_layer = tf.keras.layers.Conv1D(filters=nb_filters,
                                                       kernel_size=2,
                                                       strides=1,
                                                       padding='causal',
                                                       dilation_rate=nb_dilation_factors[i],
                                                       activation=activation,
                                                       kernel_regularizer=regularizer,
                                                       kernel_initializer=initializer,
                                                       kernel_constraint=None)
                    #Residual Connections
                    carry = tf.keras.layers.add([leaky_relu(dcc_layer(carry)), carry])

            with tf.name_scope('Final-Layer'):
                final_dcc_layer = tf.keras.layers.Conv1D(filters=1,
                                                         kernel_size=1,
                                                         strides=1,
                                                         padding='same',
                                                         activation=activation,
                                                         kernel_regularizer=regularizer,
                                                         kernel_initializer=initializer,
                                                         kernel_constraint=None)
                pred_y = leaky_relu(final_dcc_layer(carry))
            summaries = []
            with tf.name_scope('Loss'):
                loss = tf.math.reduce_mean(tf.keras.losses.mean_absolute_error(pred_y, data_y))
                naive_loss = tf.math.reduce_mean(tf.keras.losses.mean_absolute_error(data_X, data_y))
                summaries.append(tf.summary.scalar('loss', loss))
                summaries.append(tf.summary.scalar('naive_loss', naive_loss))

            return loss, pred_y, summaries

        #Conditional WaveNet
        elif conditional:
            carry = data_X
            carries = []
            for i in range(nb_input_features):
                with tf.name_scope('Conditional-skip-connections'):
                    carry_i = tf.expand_dims(carry[:, :, i], axis=-1)
                    # Skip connection
                    skip_connection_i = tf.keras.layers.Conv1D(filters=nb_filters,
                                                               kernel_size=1,
                                                               padding='same',
                                                               activation=activation,
                                                               kernel_regularizer=regularizer,
                                                               kernel_initializer=initializer)
                    dcc_layer1_i = tf.keras.layers.Conv1D(filters=nb_filters,
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
            for i in range(nb_layers):
                with tf.name_scope('Dilated-Stack'):
                    dcc_layer = tf.keras.layers.Conv1D(filters=nb_filters,
                                                       kernel_size=2,
                                                       strides=1,
                                                       padding='causal',
                                                       dilation_rate=nb_dilation_factors[i],
                                                       activation=activation,
                                                       kernel_regularizer=regularizer,
                                                       kernel_initializer=initializer)
                    # Residual Connections
                    carry = tf.keras.layers.add([leaky_relu(dcc_layer(carry)), carry])
            with tf.name_scope('Final-Layer'):
                final_dcc_layer = tf.keras.layers.Conv1D(filters=1,
                                                         kernel_size=1,
                                                         strides=1,
                                                         padding='same',
                                                         activation=activation,
                                                         kernel_regularizer=regularizer,
                                                         kernel_initializer=initializer)

                pred_y = leaky_relu(final_dcc_layer(carry))

            summaries = []
            with tf.name_scope('Loss'):
                loss = tf.math.reduce_mean(tf.keras.losses.mean_absolute_error(pred_y, data_y))
                naive_loss = tf.math.reduce_mean(tf.keras.losses.mean_absolute_error(data_X, data_y))
                summaries.append(tf.summary.scalar('loss', loss))
                summaries.append(tf.summary.scalar('naive_loss', naive_loss))

            return loss, pred_y, summaries


    def fit(self,
            forecast_data: ForecastTimeSeries,
            epochs: int):

        try:
            shutil.rmtree(self.model_path)
            shutil.rmtree(self.model_path)
        except FileNotFoundError as fnf_error:
            pass
        tf.random.set_random_seed(22943)

        self.saver = tf.train.Saver()
        with tf.Session() as sess:
            pprint(self.model_json)

            for name, graph_elements in self.model_json.items():
                print(name)
                sess.run(tf.variables_initializer(tf.global_variables(scope=name)))

                train_writer = tf.summary.FileWriter(self.model_path + '/logs/' + name + '/train', sess.graph)
                test_writer = tf.summary.FileWriter(self.model_path + '/logs/' + name + '/test')

                train_i = 0
                val_i = 0

                train_op = self.model_json[name]['train_op']
                loss = self.model_json[name]['loss']
                pred_y = self.model_json[name]['pred_y']
                target_idx = self.model_json[name]['index']
                summaries = self.model_json[name]['summaries']
                merged = tf.summary.merge(summaries)

                if self.conditional:
                    train_X = forecast_data.reshaped_rolling['Train']['features']
                else:
                    train_X = forecast_data.reshaped_rolling['Train']['features'][:, :, target_idx]
                    train_X = np.expand_dims(train_X, axis=CHANNEL_INDEX)
                train_y = forecast_data.reshaped_rolling['Train']['targets'][:, :, target_idx]
                train_y = np.expand_dims(train_y, axis=CHANNEL_INDEX)

                if self.conditional:
                    val_X = forecast_data.reshaped_rolling['Validation']['features']
                else:
                    val_X = forecast_data.reshaped_rolling['Validation']['features'][:, :, target_idx]
                    val_X = np.expand_dims(val_X, axis=CHANNEL_INDEX)
                val_y = forecast_data.reshaped_rolling['Validation']['targets'][:, :, target_idx]
                val_y = np.expand_dims(val_y, axis=CHANNEL_INDEX)


                for _ in tqdm(range(epochs)):
                    sess.run(self.iterator.initializer,
                             feed_dict={self.placeholder_X: train_X,
                                        self.placeholder_y: train_y})
                    while (True):
                        try:
                            _, eloss, epred_y, edata_y, esummary = sess.run([train_op, loss, pred_y, self.data_y, merged])
                            train_writer.add_summary(esummary, train_i)
                            train_i += 1
                        except tf.errors.OutOfRangeError:
                            break

                    sess.run(self.iterator.initializer,
                             feed_dict={self.placeholder_X: val_X,
                                        self.placeholder_y: val_y})

                    while (True):
                        try:
                            eloss, epred_y, edata_y, esummary = sess.run([loss, pred_y, self.data_y, merged])
                            test_writer.add_summary(esummary, val_i)
                            val_i += 1
                        except tf.errors.OutOfRangeError:
                            break


            self.saver.save(sess, self.model_path + '/' + self.name, global_step=epochs)
            self.meta_path = self.model_path + '/' + self.name + '-%d.meta' % epochs



    def evaluate(self,
                forecast_data: ForecastTimeSeries) -> np.array:
        nb_steps_out = forecast_data.nb_steps_out
        test_periods = forecast_data.reshaped_periods['Test']

        if self.conditional:
            with tf.Session() as sess:
                new_saver = tf.train.import_meta_graph(self.meta_path)
                new_saver.restore(sess, tf.train.latest_checkpoint(self.model_path))
                predictions = np.zeros((test_periods['targets'].shape[BATCH_INDEX],
                                        test_periods['targets'].shape[TIME_INDEX],
                                        self.nb_input_features))
                test_X = test_periods['features']
                future = test_periods['targets']
                dates = np.array(test_periods['dates'])
                carry = test_X
                for i in range(nb_steps_out):
                    for name, graph_elements in self.model_json.items():
                        pred_y = self.model_json[name]['pred_y']
                        target_idx = self.model_json[name]['index']

                        proxy_y = np.zeros((carry.shape[BATCH_INDEX], carry.shape[TIME_INDEX], 1))


                        sess.run(self.iterator.initializer,
                                 feed_dict={self.placeholder_X: carry, self.placeholder_y: proxy_y})
                        next_steps = []
                        while (True):
                            try:
                                next_step = sess.run([pred_y])[0]
                                next_steps.append(next_step)
                            except tf.errors.OutOfRangeError:
                                break
                        next_steps = np.expand_dims(np.vstack(next_steps)[:, -1, :], axis=TIME_INDEX)
                        predictions[:, i, target_idx] = next_steps.flatten()

                    predict_i = np.expand_dims(predictions[:, i, :], axis=TIME_INDEX)
                    carry = np.concatenate([carry, predict_i], axis=TIME_INDEX)
                    carry = carry[:, 1:, :]
            self.plot(predictions[:, :, 0], future[:, :, 0], dates)
            return predictions, future, dates


        else:
            with tf.Session() as sess:
                new_saver = tf.train.import_meta_graph(self.meta_path)
                new_saver.restore(sess, tf.train.latest_checkpoint(self.model_path))
                for name, graph_elements in self.model_json.items():
                    pred_y = self.model_json[name]['pred_y']
                    target_idx = self.model_json[name]['index']

                    test_X = test_periods['features'][:, :, target_idx]
                    test_X = np.expand_dims(test_X, axis=CHANNEL_INDEX)
                    test_y = test_periods['targets'][:, :, target_idx]
                    test_y = np.expand_dims(test_y, axis=CHANNEL_INDEX)
                    proxy_y = np.zeros((test_X.shape[BATCH_INDEX], test_X.shape[TIME_INDEX], 1))
                    dates = test_periods['dates']
                    carry = test_X
                    step_predictions = []
                    for i in range(nb_steps_out):
                        sess.run(self.iterator.initializer, feed_dict={self.placeholder_X: carry, self.placeholder_y: proxy_y})
                        next_steps = []
                        while (True):
                            try:
                                next_step = sess.run([pred_y])[0]
                                next_steps.append(next_step)
                            except tf.errors.OutOfRangeError:
                                break
                        next_steps = np.expand_dims(np.vstack(next_steps)[:, -1, :], axis=TIME_INDEX)
                        step_predictions.append(next_steps)
                        carry = np.concatenate([carry, next_steps], axis=TIME_INDEX)
                        carry = carry[:, 1:, :]

                    dates = np.hstack(dates)
                    predictions = np.concatenate(step_predictions, axis=TIME_INDEX)
                    self.plot(predictions, test_y, dates)
                    mape, mase, rmse = generate_stats(predictions, test_y)



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

    def plot(self, predictions, future, dates, nb_display=5):
        nb_steps_out = future.shape[TIME_INDEX]
        fig, ax = plt.subplots()
        predictions = predictions.flatten()
        future = future.flatten()
        sns.lineplot(dates, predictions, label='prediction', ax=ax)
        sns.lineplot(dates, future, label='actual', ax=ax)
        predict_dates = dates[0::nb_steps_out]
        ax.scatter(predict_dates, np.zeros(predict_dates.shape), s=30, c='r')
        ax.grid()
        plt.grid()
        plt.show()

def main():
   pass


if __name__ == '__main__':
    main()