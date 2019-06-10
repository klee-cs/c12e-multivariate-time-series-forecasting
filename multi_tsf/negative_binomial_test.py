import numpy as np
import pandas as pd
import tensorflow_probability as tfp
import tensorflow as tf
tfd = tfp.distributions
import matplotlib.pyplot as plt
import seaborn as sns

BATCH_INDEX = 0
TIME_INDEX = 1
CHANNEL_INDEX = 2

def wavenet_model_fn(features: tf.Tensor,
                     params: dict):

    nb_dilation_factors = [2**i for i in range(0, params['nb_layers'])]
    nb_filters = params['nb_filters']
    nb_layers = len(nb_dilation_factors)
    regularizer = tf.keras.regularizers.l2(l=params['l2_regularization'])
    initializer = tf.keras.initializers.he_normal()
    activation = tf.keras.activations.linear
    leaky_relu = tf.keras.layers.LeakyReLU()
    carry = features
    v0 = 1.0 + tf.cumsum(carry, axis=TIME_INDEX) / tf.reshape(tf.range(1, tf.cast(carry.shape[TIME_INDEX]+1, tf.float64)), (1, -1, 1))
    # carry = carry / v0
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


    with tf.name_scope('NegBinomial-likelihood'):
        tc_layer = tf.keras.layers.Conv1D(filters=1,
                                          kernel_size=2,
                                          padding='causal',
                                          use_bias=True,
                                          dilation_rate=nb_dilation_factors[-1],
                                          activation=activation,
                                          kernel_regularizer=regularizer,
                                          kernel_initializer=initializer,
                                          kernel_constraint=None)

        logits_layer = tf.keras.layers.Conv1D(filters=1,
                                                kernel_size=2,
                                                padding='causal',
                                                use_bias=True,
                                                dilation_rate=nb_dilation_factors[-1],
                                                activation=activation,
                                                kernel_regularizer=regularizer,
                                                kernel_initializer=initializer,
                                                kernel_constraint=None)


        total_count = tc_layer(carry)
        logits = logits_layer(carry)

        nbinomial = tfd.NegativeBinomial(total_count=total_count, logits=logits)

        ll = nbinomial.log_prob(features)
        samples = nbinomial.sample(params['num_samples'])

    return total_count, logits, ll, samples

if __name__ == '__main__':
    ts_df = pd.read_csv('./data/top_volume_active_work_sets.csv', index_col=0)
    y = ts_df.iloc[:, 0].values.reshape(1, -1, 1)
    length = y.shape[0]

    x = tf.constant(y)

    total_count, logits, ll, samples = wavenet_model_fn(x,  params={
                                         'nb_filters': 16,
                                         'nb_layers': 8,
                                         'learning_rate': 1e-3,
                                         'l2_regularization': 0.1,
                                         'num_samples': 1
                                     })
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        total_count, logits, ll, samples = sess.run([total_count, logits, ll, samples])


    mean_samples = np.mean(samples, axis=0).reshape(-1, )
    y = y.reshape(-1, )
    total_count = total_count.reshape(-1, )
    logits = logits.reshape(-1, )
    ll = ll.reshape(-1, )

    explosions = np.where(-ll < -10000)
    non_explosions = np.where(-ll > -10000)


    fig, ax = plt.subplots(3, 1)
    sns.distplot(total_count[explosions], ax=ax[0], kde=False, label='total_count_ex')
    sns.distplot(total_count[non_explosions], ax=ax[0], kde=False, label='total_count_nex')
    sns.distplot(logits[explosions], ax=ax[1], kde=False, label='logits_ex')
    sns.distplot(logits[non_explosions], ax=ax[1], kde=False, label='logits_nex')
    sns.distplot(y[explosions], ax=ax[2], kde=False, label='data_ex')
    sns.distplot(y[non_explosions], ax=ax[2], kde=False, label='data_nex')
    plt.legend()
    plt.show()

    print(np.mean(ll))
    #
    # all_stats = np.vstack([y, log_rate, ll, mean_samples])


