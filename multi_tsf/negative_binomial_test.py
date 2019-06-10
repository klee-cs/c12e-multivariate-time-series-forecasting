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
    carry = carry / v0
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
        logits = tf.abs(1 + logits_layer(carry))

        normal = tfd.Normal(loc=total_count, scale=logits)

        nbinomial = tfd.NegativeBinomial(total_count=total_count,
                                         logits=logits)

        ll = normal.log_prob(features)

        samples = normal.sample(params['num_samples'])

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
                                         'num_samples': 100
                                     })
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        tc, logits, ll, samples = sess.run([total_count, logits, ll, samples])


    mean_samples = np.mean(samples, axis=0).reshape(-1, )
    y = y.reshape(-1, )
    tc = tc.reshape(-1, )
    logits = logits.reshape(-1, )
    ll = ll.reshape(-1, )



    print(y[0:150])
    print(tc[0:150])
    print(logits[0:150])
    print(ll[0:150])
    print(mean_samples[0:150])
    print(np.mean(ll))

    fig, ax = plt.subplots(5, 1)
    ax[0].plot(y[0:150])
    ax[1].plot(tc[0:150])
    ax[2].plot(logits[0:150])
    ax[3].plot(ll[0:150])
    ax[4].plot(mean_samples[0:150])
    plt.show()

    all_stats = np.vstack([y, tc, logits, ll, mean_samples])


    # total_count = tfd.Gamma(concentration=1.0, rate=2.0).sample(length)
    # logits = tfd.Normal(loc=0.0, scale=1.0).sample(length)
    # nb = tfd.NegativeBinomial(total_count=2.1435, logits=0.0)
    # sess = tf.Session()
    # print(sess.run(tf.reduce_mean(nb.log_prob(2.0))))
    # sns.distplot(sess.run(nb.log_prob(y)))
    # plt.show()