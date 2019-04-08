import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from multi_tsf.time_series_utils import SyntheticSinusoids, ForecastTimeSeries
from typing import List

class WaveNet_Forecasting_Model(object):

    def __init__(self,
                 nb_layers: int,
                 nb_filters: int,
                 nb_dilation_factors: List[int],
                 nb_steps_in: int,
                 nb_input_features: int,
                 nb_output_features: int) -> None:
        self.nb_layers = nb_layers
        self.nb_filters = nb_filters
        self.nb_dilation_factors = nb_dilation_factors
        self.nb_steps_in = nb_steps_in
        self.nb_input_features = nb_input_features
        self.nb_output_features = nb_output_features

    def __create_model(self) -> None:
        carry = self.data_X
        for i in range(self.nb_layers):
            dcc_layer = keras.layers.Conv1D(filters=self.nb_filters,
                                            kernel_size=2,
                                            strides=1,
                                            padding='causal',
                                            dilation_rate=self.nb_dilation_factors[i],
                                            activation=keras.activations.relu)
            carry = dcc_layer(carry)

    def fit(self,
            forecast_data: ForecastTimeSeries,
            epochs: int,
            batch_size: int = 64,
            lr: float = 1e-3) -> None:
        self.batch_size = batch_size
        self.placeholder_X = tf.placeholder(tf.float32, [None, self.nb_steps_in, self.nb_input_features])
        self.placeholder_y = tf.placeholder(tf.float32, [None, self.nb_steps_in, self.nb_output_features])


def main():
    pass

if __name__ == '__main__':
    main()