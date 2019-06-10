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
    carry = features

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
    for i in range(nb_layers-1):
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
                                                 kernel_size=2,
                                                 strides=1,
                                                 padding='causal',
                                                 dilation_rate=nb_dilation_factors[-1],
                                                 activation=activation,
                                                 kernel_regularizer=regularizer,
                                                 kernel_initializer=initializer,
                                                 kernel_constraint=None)
        pred_y = leaky_relu(final_dcc_layer(carry))




    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'pred_y': pred_y
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.math.reduce_mean(tf.keras.losses.mean_absolute_error(pred_y, labels), name='loss')
        naive_loss = tf.math.reduce_mean(tf.keras.losses.mean_absolute_error(features, labels))
        tf.summary.scalar(name='MAE', tensor=loss)
        tf.summary.scalar(name='Naive-MAE', tensor=naive_loss)
        optimizer = tf.train.AdamOptimizer(params['learning_rate'], name='optimizer')
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        logging_hook = tf.train.LoggingTensorHook({"loss": loss}, every_n_iter=1)

        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])


    # if mode == tf.estimator.ModeKeys.TRAIN:
    #
    #     ll = normal.log_prob(labels)
    #     loss = -tf.reduce_mean(ll, name='loss')
    #     tf.summary.histogram(name='v', values=v0)
    #     tf.summary.histogram(name='total_count', values=total_count)
    #     tf.summary.histogram(name='logits', values=logits)
    #     tf.summary.scalar(name='nll', tensor=loss)
    #     optimizer = tf.train.AdamOptimizer(params['learning_rate'], name='optimizer')
    #     train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    #
    #     logging_hook = tf.train.LoggingTensorHook({"total_count": tf.reduce_mean(total_count),
    #                                                "logits": tf.reduce_mean(logits),
    #                                                "loss": loss}, every_n_iter=1)
    #
    #     return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])



def input_fn(path: str,
             target_index: int,
             forecast_horizon: int,
             num_epochs: Optional[int],
             mode: str):
    ts_df = pd.read_csv(path, index_col=0)
    time_series = ts_df.values
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
    target_index = 0
    num_samples = 100
    test_y = test_df.iloc[forecast_horizon:, target_index].values

    wavenet = tf.estimator.Estimator(model_fn=wavenet_model_fn,
                                     model_dir=model_save_path,
                                     params={
                                         'nb_filters': 16,
                                         'nb_layers': 8,
                                         'learning_rate': 1e-3,
                                         'l2_regularization': 0.1,
                                         'num_samples': num_samples,
                                         'MAE_loss': True
                                     })



    wavenet.train(input_fn=lambda: input_fn(path='./data/train.csv',
                                            target_index=target_index,
                                            forecast_horizon=forecast_horizon,
                                            num_epochs=1000,
                                            mode=tf.estimator.ModeKeys.TRAIN))

    predictions = wavenet.predict(input_fn=lambda: input_fn(path='./data/test.csv',
                                                            target_index=target_index,
                                                            forecast_horizon=forecast_horizon,
                                                            num_epochs=None,
                                                            mode=tf.estimator.ModeKeys.PREDICT))


    idx = np.arange(0, test_y.shape[0])
    pred_y = np.array([p['pred_y'] for p in predictions]).reshape(-1, )
    plt.plot(test_y)
    plt.plot(pred_y)
    plt.show()



    # # pred_y[pred_y > 1000] = 0
    # mean_y = np.mean(pred_y, axis=0)
    # low_percentile = np.percentile(pred_y, q=2.5, axis=0).reshape(-1,)
    # high_percentile = np.percentile(pred_y, q=97.5, axis=0).reshape(-1,)
    # sample_mean = np.mean(pred_y, axis=0).reshape(-1,)
    # fig, ax = plt.subplots()
    # ax.fill_between(idx,
    #                 low_percentile,
    #                 high_percentile,
    #                 alpha=0.1, color='b')
    # ax.plot(idx, test_y, label='actual')
    # ax.plot(idx, sample_mean, label='predicted')
    # ax.legend()
    # plt.show()

