import numpy as np
import pandas as pd
import os
import shutil
import tensorflow as tf
from multi_tsf.time_series_utils import ForecastTimeSeries
from typing import List
from pprint import pprint
from tqdm import tqdm
import json

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
                 ts_names: dict,
                 from_file_model_params: dict=None) -> None:

        self.ts_names = [x.replace(' ', '-') for x in ts_names]
        self.train_model_params = {}
        self.model_path = model_path
        self.name = name
        self.conditional = conditional
        self.nb_layers = nb_layers
        self.nb_filters = nb_filters
        self.nb_dilation_factors = nb_dilation_factors
        if self.conditional:
            self.nb_input_features = nb_input_features
        else:
            self.nb_input_features = 1
        os.makedirs(model_path, exist_ok=True)
        self.batch_size = batch_size


        if from_file_model_params is None:
            self.placeholder_X = tf.placeholder(tf.float32, [None, None, self.nb_input_features])
            self.placeholder_y = tf.placeholder(tf.float32, [None, None, 1])
            self.dataset = tf.data.Dataset.from_tensor_slices((self.placeholder_X, self.placeholder_y))
            self.batched_dataset = self.dataset.batch(batch_size=self.batch_size)
            self.iterator = self.batched_dataset.make_initializable_iterator()
            self.init_op = self.iterator.make_initializer(self.batched_dataset, name='iterator_init_op')
            self.data_X, self.data_y = self.iterator.get_next()

            #Create models for each time series
            self.model_params = {
                'ts_names': self.ts_names,
                'nb_layers': nb_layers,
                'nb_filters': nb_filters,
                'nb_dilation_factors': nb_dilation_factors,
                'nb_input_features': nb_input_features,
                'lr': lr,
                'model_path': self.model_path,
                'name': self.name,
                'conditional': self.conditional,
                'batch_size': self.batch_size,
                'placeholder_X': self.placeholder_X.name,
                'placeholder_y': self.placeholder_y.name,
                'iterator': self.init_op.name,
                'predict_tensor_names': {}
            }


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
                        optimizer = tf.train.AdamOptimizer(lr, name='optimizer')
                        gradients = optimizer.compute_gradients(loss)

                        for gradient, variable in gradients:
                            if gradient is None or variable is None:
                                continue
                            summaries.append(tf.summary.histogram("gradients/" + variable.name, gradient))
                            summaries.append(tf.summary.histogram("variables/" + variable.name, variable))

                        train_op = optimizer.minimize(loss)

                        self.model_params['predict_tensor_names'][name] = {
                            'index': idx,
                            'loss': loss.name,
                            'pred_y': pred_y.name
                        }


                        train_graph_elements = {
                            'index': idx,
                            'loss': loss.name,
                            'pred_y': pred_y.name,
                            'optimizer': optimizer,
                            'train_op': train_op,
                            'summaries': summaries
                        }

                        self.train_model_params[name] = train_graph_elements

            pprint(self.model_params)
        else:
            self.model_params = from_file_model_params
            self.meta_path = from_file_model_params['meta_path']
            pprint(self.model_params)


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
                pred_y = leaky_relu(final_dcc_layer(carry), name='pred_y')
            summaries = []
            with tf.name_scope('Loss'):
                loss = tf.math.reduce_mean(tf.keras.losses.mean_absolute_error(pred_y, data_y), name='loss')
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
            for name, graph_elements in self.train_model_params.items():
                print(name)
                sess.run(tf.variables_initializer(tf.global_variables(scope=name)))

                train_writer = tf.summary.FileWriter(self.model_path + '/logs/' + name + '/train', sess.graph)
                test_writer = tf.summary.FileWriter(self.model_path + '/logs/' + name + '/test')

                train_i = 0
                val_i = 0

                train_op = self.train_model_params[name]['train_op']
                loss = tf.get_default_graph().get_tensor_by_name(self.train_model_params[name]['loss'])
                pred_y = tf.get_default_graph().get_tensor_by_name(self.train_model_params[name]['pred_y'])
                target_idx = self.train_model_params[name]['index']
                summaries = self.train_model_params[name]['summaries']
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
                    sess.run(self.init_op,
                             feed_dict={self.placeholder_X: train_X,
                                        self.placeholder_y: train_y})
                    while (True):
                        try:
                            _, eloss, epred_y, edata_y, esummary = sess.run([train_op, loss, pred_y, self.data_y, merged])
                            # if train_i % 1000 == 0:
                            #     train_writer.add_summary(esummary, train_i)
                            #     train_i += 1
                        except tf.errors.OutOfRangeError:
                            break


                    # sess.run(self.init_op,
                    #          feed_dict={self.placeholder_X: val_X,
                    #                     self.placeholder_y: val_y})
                    #
                    # while (True):
                    #     try:
                    #         eloss, epred_y, edata_y, esummary = sess.run([loss, pred_y, self.data_y, merged])
                    #         # if val_i % 1000 == 0:
                    #         #     test_writer.add_summary(esummary, val_i)
                    #         #     val_i += 1
                    #     except tf.errors.OutOfRangeError:
                    #         break



            self.meta_path = self.model_path + '/' + self.name + '-%d.meta' % epochs
            self.model_params['meta_path'] = self.meta_path
            self.save_model(sess,
                            self.model_params,
                            self.saver,
                            epochs)




    def evaluate(self,
                 forecast_data: ForecastTimeSeries,
                 set: str) -> np.array:

        new_saver = tf.train.import_meta_graph(self.meta_path)
        graph = tf.get_default_graph()
        nb_steps_out = forecast_data.nb_steps_out
        test_periods = forecast_data.reshaped_periods[set]
        placeholder_X = graph.get_tensor_by_name(self.model_params['placeholder_X'])
        placeholder_y = graph.get_tensor_by_name(self.model_params['placeholder_y'])
        init_op = graph.get_operation_by_name(self.model_params['iterator'])

        if self.conditional:
            with tf.Session() as sess:
                new_saver.restore(sess, tf.train.latest_checkpoint(self.model_path))
                future = test_periods['targets']
                predictions = np.zeros((future.shape[BATCH_INDEX],
                                        future.shape[TIME_INDEX],
                                        future.shape[CHANNEL_INDEX]))
                test_X = test_periods['features']
                dates = np.array(test_periods['dates'])
                carry = test_X
                for i in range(nb_steps_out):
                    for name, graph_elements in self.model_params['predict_tensor_names'].items():
                        pred_y = graph.get_tensor_by_name(graph_elements['pred_y'])
                        target_idx = graph_elements['index']

                        proxy_y = np.zeros((carry.shape[BATCH_INDEX], carry.shape[TIME_INDEX], 1))

                        sess.run(init_op,
                                 feed_dict={placeholder_X: carry, placeholder_y: proxy_y})
                        next_steps = []
                        while (True):
                            try:
                                next_step = sess.run([pred_y])[0]
                                next_steps.append(next_step)
                            except tf.errors.OutOfRangeError:
                                break
                        next_steps = np.vstack(next_steps)
                        next_steps = np.expand_dims(next_steps[:, -1, :], axis=TIME_INDEX)
                        predictions[:, i, target_idx] = next_steps.flatten()

                    predict_i = np.expand_dims(predictions[:, i, :], axis=TIME_INDEX)
                    carry = np.concatenate([carry, predict_i], axis=TIME_INDEX)
                    carry = carry[:, 1:, :]

            result_dict = {}
            for idx, name in enumerate(self.ts_names):
                result_dict[(name, 'prediction')] = predictions[:, :, idx].flatten()
                result_dict[(name, 'actual')] = future[:, :, idx].flatten()

            results_df = pd.DataFrame(result_dict)
            results_df.index = dates
            results_df.T.to_csv(self.model_path + '/results_df.csv', index_label=['first', 'second'])

            return results_df


        else:
            with tf.Session() as sess:
                new_saver.restore(sess, tf.train.latest_checkpoint(self.model_path))
                future = test_periods['targets']
                predictions = np.zeros((future.shape[BATCH_INDEX],
                                        future.shape[TIME_INDEX],
                                        test_periods['features'].shape[CHANNEL_INDEX]))
                dates = np.array(test_periods['dates'])
                for name, graph_elements in self.model_params['predict_tensor_names'].items():
                    pred_y = graph.get_tensor_by_name(graph_elements[name]['pred_y'])
                    target_idx = graph_elements['index']

                    test_X = test_periods['features'][:, :, target_idx]
                    test_X = np.expand_dims(test_X, axis=CHANNEL_INDEX)
                    proxy_y = np.zeros((test_X.shape[BATCH_INDEX], test_X.shape[TIME_INDEX], 1))

                    carry = test_X
                    step_predictions = []
                    for i in range(nb_steps_out):
                        sess.run(init_op, feed_dict={placeholder_X: carry, placeholder_y: proxy_y})
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

                    predict_name = np.concatenate(step_predictions, axis=TIME_INDEX)
                    predict_name = np.squeeze(predict_name)
                    predictions[:, :, target_idx] = predict_name

            result_dict = {}
            for idx, name in enumerate(self.ts_names):
                result_dict[(name, 'prediction')] = predictions[:, :, idx].flatten()
                result_dict[(name, 'actual')] = future[:, :, idx].flatten()

            results_df = pd.DataFrame(result_dict)
            results_df.index = dates
            results_df.T.to_csv(self.model_path + '/results_df.csv', index_label=['first', 'second'])

            return results_df

    #TODO write predict for a single batch without ground truth
    def predict(self,
                time_series: np.array,
                nb_steps_out: int):
        future = np.zeros((1, time_series.shape[TIME_INDEX], time_series.shape[CHANNEL_INDEX]))
        test_X = np.expand_dims(time_series, axis=BATCH_INDEX)


    def save_model(self,
                       sess: tf.Session,
                       model_params: dict,
                       saver: tf.train.Saver,
                       epochs: int) -> None:
            saver.save(sess, model_params['model_path'] + '/' + self.name, global_step=epochs)
            with open(model_params['model_path'] + '/model_params.json', 'w+') as f:
                json.dump(model_params, f)


    @staticmethod
    def restore_model(model_params_path: str):
        with open(model_params_path, 'r') as f:
            model_params = json.load(f)
            wavenet_forecasting_model = WaveNetForecastingModel(name=model_params['name'],
                                                                conditional=model_params['conditional'],
                                                                nb_layers=model_params['nb_layers'],
                                                                nb_filters=model_params['nb_filters'],
                                                                nb_dilation_factors=model_params['nb_dilation_factors'],
                                                                nb_input_features=model_params['nb_input_features'],
                                                                batch_size=model_params['batch_size'],
                                                                lr=model_params['lr'],
                                                                model_path=model_params['model_path'],
                                                                ts_names=model_params['ts_names'],
                                                                from_file_model_params=model_params)
        return wavenet_forecasting_model




