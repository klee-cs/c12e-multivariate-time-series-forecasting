import numpy as np
import pandas as pd
import os
import shutil
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from multi_tsf.time_series_utils import ForecastTimeSeries, train_val_test_split
from typing import List
from pprint import pprint
from tqdm import tqdm
import json
from typing import Optional
import matplotlib.pyplot as plt

BATCH_INDEX = 0
TIME_INDEX = 1
CHANNEL_INDEX = 2

def wavenet_model_fn(features: tf.Tensor,
                     labels: tf.Tensor,
                     mode: str,
                     params: dict):

    nb_dilation_factors = [2**i for i in range(0, params['nb_layers'])]
    nb_filters = params['nb_filters']
    nb_layers = len(nb_dilation_factors)
    regularizer = tf.keras.regularizers.l2(l=params['l2_regularization'])
    initializer = tf.keras.initializers.he_normal()
    activation = tf.keras.activations.linear
    leaky_relu = tf.keras.layers.LeakyReLU()
    nb_input_features = features.get_shape().as_list()[CHANNEL_INDEX]
    carry = features

    if params['MAE_loss'] == True:
        v0 = 0.0
    elif params['MAE_loss'] == False:
        batch_size = carry.get_shape().as_list()[BATCH_INDEX]
        time_steps = carry.get_shape().as_list()[TIME_INDEX]
        num_channels = carry.get_shape().as_list()[CHANNEL_INDEX]
        v0 = 1.0 + tf.cumsum(carry, axis=TIME_INDEX) / tf.reshape(tf.range(1.0, tf.cast(time_steps + 1.0, tf.float64)), (1, -1, num_channels))
        carry = carry / v0


    if params['conditional']:
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

    else:
        # Skip connection
        skip_connection = tf.keras.layers.Conv1D(filters=nb_filters,
                                                 kernel_size=1,
                                                 padding='same',
                                                 use_bias=True,
                                                 activation=activation,
                                                 kernel_regularizer=regularizer,
                                                 kernel_initializer=initializer,
                                                 kernel_constraint=None)

        dcc_layer1 = tf.keras.layers.Conv1D(filters=nb_filters,
                                            kernel_size=2,
                                            strides=1,
                                            padding='causal',
                                            use_bias=True,
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
                                               use_bias=True,
                                               dilation_rate=nb_dilation_factors[i],
                                               activation=activation,
                                               kernel_regularizer=regularizer,
                                               kernel_initializer=initializer,
                                               kernel_constraint=None)
            # Residual Connections
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

    with tf.name_scope('Negative-Binomial-Likelihood'):
        tc_layer = tf.keras.layers.Conv1D(filters=1,
                                          kernel_size=1,
                                          padding='same',
                                          use_bias=True,
                                          activation=activation,
                                          kernel_regularizer=regularizer,
                                          kernel_initializer=initializer,
                                          kernel_constraint=None)

        logits_layer = tf.keras.layers.Conv1D(filters=1,
                                              kernel_size=1,
                                              padding='same',
                                              use_bias=True,
                                              activation=activation,
                                              kernel_regularizer=regularizer,
                                              kernel_initializer=initializer,
                                              kernel_constraint=None)

        constant_one = tf.constant(1., dtype=tf.float64)
        total_count = v0 * tf.math.maximum(constant_one, tc_layer(carry))
        logits = logits_layer(carry)
        nbinomial = tfd.NegativeBinomial(total_count=total_count, logits=logits)
        ll = nbinomial.log_prob(features)
        samples = nbinomial.sample(params['num_samples'])


    if mode == tf.estimator.ModeKeys.PREDICT:
        if params['MAE_loss'] == True:
            predictions = {
                'pred_y': pred_y
            }
        elif params['MAE_loss'] == False:
            predictions = {
                'pred_y': samples
            }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    if mode == tf.estimator.ModeKeys.TRAIN:
        if params['MAE_loss'] == True:
            loss = tf.math.reduce_mean(tf.keras.losses.mean_absolute_error(pred_y, labels), name='loss')
            naive_loss = tf.math.reduce_mean(tf.keras.losses.mean_absolute_error(features, labels))
            tf.summary.scalar(name='MAE', tensor=loss)
            tf.summary.scalar(name='Naive-MAE', tensor=naive_loss)
            optimizer = tf.train.AdamOptimizer(params['learning_rate'], name='optimizer')
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

            logging_hook = tf.train.LoggingTensorHook({"loss": loss}, every_n_iter=1)

            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])

        elif params['MAE_loss'] == False:
            ll = nbinomial.log_prob(labels)
            loss = -tf.reduce_mean(ll, name='loss')
            tf.summary.histogram(name='total_count', values=total_count)
            tf.summary.histogram(name='logits', values=logits)
            tf.summary.scalar(name='nll', tensor=loss)
            optimizer = tf.train.AdamOptimizer(params['learning_rate'], name='optimizer')
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

            logging_hook = tf.train.LoggingTensorHook({"total_count": tf.reduce_mean(total_count),
                                                       "logits": tf.reduce_mean(logits),
                                                       "loss": loss}, every_n_iter=1)

            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])


def input_fn(path: str,
             target_index: int,
             forecast_horizon: int,
             num_epochs: Optional[int],
             conditional: bool,
             mode: str):
    ts_df = pd.read_csv(path, index_col=0)
    time_series = ts_df.values

    if conditional:
        seq_x = time_series[:-forecast_horizon, :]
        seq_y = time_series[forecast_horizon:, target_index].reshape(-1, 1)
    else:
        seq_x = time_series[:-forecast_horizon, target_index].reshape(-1, 1)
        seq_y = time_series[forecast_horizon:, target_index].reshape(-1, 1)

    seq_x = np.expand_dims(seq_x, axis=0)
    seq_y = np.expand_dims(seq_y, axis=0)
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = tf.data.Dataset.from_tensor_slices((seq_x, seq_y)).repeat(num_epochs).batch(batch_size=1)
    elif mode == tf.estimator.ModeKeys.PREDICT:
        dataset = tf.data.Dataset.from_tensor_slices((seq_x, seq_y)).batch(batch_size=1)
    return dataset


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    model_save_path = './estimator_test'
    try:
        shutil.rmtree(model_save_path)
    except:
        pass
    train_df, val_df, test_df = train_val_test_split(data_path='./data/top_volume_active_work_sets.csv',
                                                     save_path='./data',
                                                     test_cutoff_date='2019-01-01',
                                                     train_size=0.8)


    forecast_horizon = 34
    target_index = 20
    num_samples = 100
    num_epochs = 1000
    test_y = test_df.iloc[forecast_horizon:, target_index].values

    wavenet = tf.estimator.Estimator(model_fn=wavenet_model_fn,
                                     model_dir=model_save_path,
                                     params={
                                         'nb_filters': 16,
                                         'nb_layers': 8,
                                         'learning_rate': 1e-3,
                                         'l2_regularization': 0.1,
                                         'conditional': False,
                                         'num_samples': num_samples,
                                         'MAE_loss': False
                                     })



    wavenet.train(input_fn=lambda: input_fn(path='./data/train.csv',
                                            target_index=target_index,
                                            forecast_horizon=forecast_horizon,
                                            num_epochs=num_epochs,
                                            conditional=False,
                                            mode=tf.estimator.ModeKeys.TRAIN))

    # early_stopping = tf.estimator.stop_if_no_decrease_hook(
    #     wavenet,
    #     metric_name='loss',
    #     max_steps_without_decrease=1000,
    #     min_steps=100
    # )

    predictions = wavenet.predict(input_fn=lambda: input_fn(path='./data/test.csv',
                                                            target_index=target_index,
                                                            forecast_horizon=forecast_horizon,
                                                            num_epochs=None,
                                                            conditional=False,
                                                            mode=tf.estimator.ModeKeys.PREDICT))


    idx = np.arange(0, test_y.shape[0])
    # pred_y = np.array([p['pred_y'] for p in predictions]).reshape(-1, )
    # plt.plot(test_y)
    # plt.plot(pred_y)
    # plt.show()

    pred_y = np.array([p['pred_y'] for p in predictions]).reshape(num_samples, -1, 1)
    pred_y[pred_y > np.max(test_y)] = 0
    low_percentile = np.percentile(pred_y, q=2.5, axis=0).reshape(-1,)
    high_percentile = np.percentile(pred_y, q=97.5, axis=0).reshape(-1,)
    sample_mean = np.mean(pred_y, axis=0).reshape(-1,)
    fig, ax = plt.subplots()
    ax.fill_between(idx,
                    low_percentile,
                    high_percentile,
                    alpha=0.1, color='b')
    ax.plot(idx, test_y, label='actual')
    ax.plot(idx, sample_mean, label='predicted')
    ax.legend()
    plt.show()
    #
