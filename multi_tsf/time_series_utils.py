import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

class ForecastTimeSeries(object):

    def __init__(self,
                 time_series: np.array,
                 train_size: float,
                 val_size: float,
                 nb_steps_in,
                 nb_steps_out,
                 target_index):
        self.time_series = time_series
        if nb_steps_out is None:
            self._create_shifted_feature_targets(max_look_back=nb_steps_in,
                                                 target_index=target_index)
        else:
            self._create_lags_and_target(nb_steps_in=nb_steps_in,
                                         nb_steps_out=nb_steps_out,
                                         target_index=target_index)
        self._split_train_validation_test(train_size=train_size,
                                          val_size=val_size)


    def _create_lags_and_target(self,
                                nb_steps_in: int,
                                nb_steps_out: int,
                                target_index: int = None) -> None:

        self.nb_steps_in = nb_steps_in
        self.nb_steps_out = nb_steps_out

        X, y = [], []
        for i in range(self.time_series.shape[0]):
            # find the end of this pattern
            end_ix = i + self.nb_steps_in
            out_end_ix = end_ix + self.nb_steps_out
            # check if we are beyond the dataset
            if out_end_ix > len(self.time_series):
                break
            # gather input and output parts of the pattern
            if target_index is not None:
                seq_x, seq_y = self.time_series[i:end_ix, :], self.time_series[end_ix:out_end_ix, target_index]
                seq_y = np.expand_dims(seq_y, axis=1)
            else:
                seq_x, seq_y = self.time_series[i:end_ix, :], self.time_series[end_ix:out_end_ix, :]
            X.append(seq_x)
            y.append(seq_y)

        self.features = np.array(X)
        self.targets = np.array(y)
        self.nb_input_features = self.features.shape[-1]
        self.nb_output_features = self.features.shape[-1]


    def _create_shifted_feature_targets(self,
                                max_look_back: int,
                                target_index: int) -> None:

        X, y = [], []
        for i in range(self.time_series.shape[0]):
            # find the end of this pattern
            end_ix = i + max_look_back
            # check if we are beyond the dataset
            if end_ix > len(self.time_series):
                break
            # gather input and output parts of the pattern
            if target_index is not None:
                seq_x, seq_y = self.time_series[i:end_ix-1, :], self.time_series[i+1:end_ix, target_index]
                seq_y = np.expand_dims(seq_y, axis=1)
            else:
                seq_x, seq_y = self.time_series[i:end_ix-1, :], self.time_series[i+1:end_ix, :]
            X.append(seq_x)
            y.append(seq_y)

        self.features = np.array(X)
        self.targets = np.array(y)
        self.nb_input_features = self.features.shape[-1]
        self.nb_output_features = self.features.shape[-1]


    def _split_train_validation_test(self,
                               train_size: float,
                               val_size: float) -> None:

        train_cutoff = math.ceil(self.time_series.shape[0] * train_size)
        val_cutoff = math.ceil(self.time_series.shape[0] * (train_size + val_size))

        self.train_X = self.features[:train_cutoff]
        self.train_y = self.targets[:train_cutoff]

        self.val_X = self.features[train_cutoff:val_cutoff]
        self.val_y = self.targets[train_cutoff:val_cutoff]

        self.test_X = self.features[val_cutoff:]
        self.test_y = self.targets[val_cutoff:]



class SyntheticSinusoids(object):

    def __init__(self, num_sinusoids, amplitude, sampling_rate, length):
        self.num_sinusoids = num_sinusoids
        self.amplitude = amplitude
        self.sampling_rate = sampling_rate
        self.length = length
        n = np.arange(0, self.length)

        self.sinusoids = []
        for i in range(num_sinusoids):
            f = np.random.uniform(int(sampling_rate//100), int(self.sampling_rate//30))
            noise_level = np.random.uniform(0.01, 0.02)
            x = self.amplitude*np.cos(2*np.pi*f/self.sampling_rate*n) \
             + np.random.normal(loc=0.0, scale=noise_level*self.amplitude, size=n.shape[0])
            self.sinusoids.append(x.reshape(-1, 1))

        self.sinusoids = np.concatenate(self.sinusoids, axis=1)


    def plot(self):
        for i in range(self.sinusoids.shape[1]):
            plt.plot(self.sinusoids[:, i])
        plt.show()


def main():
    nb_steps_in = 100
    nb_steps_out = 5
    target_index = None
    train_size = 0.7
    val_size = 0.15
    synthetic_sinusoids = SyntheticSinusoids(num_sinusoids=5,
                                             amplitude=1,
                                             sampling_rate=5000,
                                             length=10000)
    wavenet_sinusoid_ts = ForecastTimeSeries(synthetic_sinusoids.sinusoids,
                                             train_size=train_size,
                                             val_size=val_size,
                                             nb_steps_in=nb_steps_in,
                                             nb_steps_out=None,
                                             target_index=None)
    index = np.arange(0, 99)
    sns.lineplot(index, wavenet_sinusoid_ts.features[0, :, 0].reshape(-1, ), label='predicted')
    sns.lineplot(index, wavenet_sinusoid_ts.targets[0, :, 0].reshape(-1, ), label='actual')
    plt.legend()
    plt.show()
    lstm_sinusoid_ts = ForecastTimeSeries(synthetic_sinusoids.sinusoids,
                                          train_size=train_size,
                                          val_size=val_size,
                                          nb_steps_in=100,
                                          nb_steps_out=5,
                                          target_index=None)


if __name__ == '__main__':
    main()







