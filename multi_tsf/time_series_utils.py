import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

class ForecastTimeSeries(object):

    def __init__(self,
                 ts_df: pd.DataFrame,
                 vector_output_mode: bool,
                 train_size: float,
                 val_size: float,
                 nb_steps_in: int,
                 nb_steps_out: int,
                 target_index: Optional[int],
                 predict_hour: int,
                 by_timestamp: bool) -> None:
        self.vector_output_mode = vector_output_mode
        self.column_names = ts_df.columns
        self.time_series = ts_df.values
        self.dates = ts_df.index.tolist()
        self.nb_steps_out = nb_steps_out
        self.nb_steps_in = nb_steps_in
        self.target_index = target_index
        self.predict_hour = predict_hour
        self.by_timestamp = by_timestamp
        self.features = None
        self.targets = None
        self.nb_input_features = None
        self.nb_output_features = None
        if self.vector_output_mode:
            self._create_lags_and_target()
        else:
            self._create_shifted_feature_targets()
        self._split_train_validation_test(train_size=train_size,
                                          val_size=val_size)
        self.periods = {'Train': None, 'Validation': None, 'Test': None}
        self._create_predict_periods()


    def _create_lags_and_target(self) -> None:

        X, y = [], []
        for i in range(self.time_series.shape[0]):
            # find the end of this pattern
            end_ix = i + self.nb_steps_in
            out_end_ix = end_ix + self.nb_steps_out
            # check if we are beyond the dataset
            if out_end_ix > len(self.time_series):
                break
            # gather input and output parts of the pattern
            if self.target_index is not None:
                seq_x, seq_y = self.time_series[i:end_ix, :], self.time_series[end_ix:out_end_ix, self.target_index]
                seq_y = np.expand_dims(seq_y, axis=1)
            else:
                seq_x, seq_y = self.time_series[i:end_ix, :], self.time_series[end_ix:out_end_ix, :]
            X.append(seq_x)
            y.append(seq_y)

        self.features = np.array(X)
        self.targets = np.array(y)
        self.nb_input_features = self.features.shape[-1]
        self.nb_output_features = self.features.shape[-1]


    def _create_shifted_feature_targets(self) -> None:
        if self.target_index is not None:
            seq_x, seq_y = self.time_series[:-1, :], self.time_series[1:, self.target_index]
            seq_y = np.expand_dims(seq_y, axis=-1)
            self.nb_input_features = self.time_series.shape[-1]
            self.nb_output_features = 1
        else:
            seq_x, seq_y = self.time_series[:-1, :], self.time_series[1:, :]
            self.nb_input_features = self.nb_output_features = self.time_series.shape[-1]

        self.features = np.expand_dims(seq_x, axis=0)
        self.targets = np.expand_dims(seq_y, axis=0)


    def _create_predict_periods(self) -> (np.array, np.array):

        train_ts = self.time_series[:self.train_cutoff]
        train_dates = self.dates[:self.train_cutoff]
        val_ts = self.time_series[self.train_cutoff:self.val_cutoff]
        val_dates = self.dates[self.train_cutoff:self.val_cutoff]
        test_ts = self.time_series[self.val_cutoff:]
        test_dates = self.dates[self.val_cutoff:]

        self.periods = {
            'Train': {
                'ts': train_ts,
                'dates': train_dates,
                'predict_dates': [],
                'features': [],
                'targets': []
            },
            'Validation': {
                'ts': val_ts,
                'dates': val_dates,
                'predict_dates': [],
                'features': [],
                'targets': []
            },
            'Test': {
                'ts': test_ts,
                'dates': test_dates,
                'predict_dates': [],
                'features': [],
                'targets': []
            }
        }


        for set, data in self.periods.items():
            if self.by_timestamp:
                for i in range(data['ts'].shape[0]):
                    hour = data['dates'][i].hour
                    min = data['dates'][i].minute
                    if hour == self.predict_hour and min == 0:
                        if (i - self.nb_steps_in) >= 0 and (i + self.nb_steps_out) < data['ts'].shape[0]:
                            if self.target_index is None:
                                feature = data['ts'][i - self.nb_steps_in:i, :]
                                target = data['ts'][i:i + self.nb_steps_out, :]
                            else:
                                feature = data['ts'][i - self.nb_steps_in:i, :]
                                target = data['ts'][i:i + self.nb_steps_out, self.target_index]
                            data['predict_dates'].append(data['dates'][i:i+self.nb_steps_out])
                            data['features'].append(feature)
                            data['targets'].append(target)
            else:
                for i in np.arange(0, data['ts'].shape[0], self.nb_steps_out):
                    if (i - self.nb_steps_in) >= 0 and (i + self.nb_steps_out) < data['ts'].shape[0]:
                        if self.target_index is None:
                            feature = data['ts'][i - self.nb_steps_in:i, :]
                            target = data['ts'][i:i + self.nb_steps_out, :]
                        else:
                            feature = data['ts'][i - self.nb_steps_in:i, :]
                            target = data['ts'][i:i + self.nb_steps_out, self.target_index]
                        data['predict_dates'].append(data['dates'][i:i + self.nb_steps_out])
                        data['features'].append(feature)
                        data['targets'].append(target)
            data['features'] = np.array(data['features'])
            data['targets'] = np.array(data['targets'])
            print(data['features'].shape)
            print(data['targets'].shape)
            # if self.nb_output_features == 1:
            #     data['features'] = np.expand_dims(data['features'], axis=2)
            #     data['targets'] = np.expand_dims(data['targets'], axis=2)



    def _split_train_validation_test(self,
                               train_size: float,
                               val_size: float) -> None:

        self.train_cutoff = math.ceil(self.time_series.shape[0] * train_size)
        self.val_cutoff = math.ceil(self.time_series.shape[0] * (train_size + val_size))

        print(self.features.shape)
        print(self.targets.shape)

        if self.vector_output_mode:
            self.train_X = self.features[:self.train_cutoff]
            self.train_y = self.targets[:self.train_cutoff]

            self.val_X = self.features[self.train_cutoff:self.val_cutoff]
            self.val_y = self.targets[self.train_cutoff:self.val_cutoff]

            self.test_X = self.features[self.val_cutoff:]
            self.test_y = self.targets[self.val_cutoff:]

        else:
            self.train_X = self.features[:, :self.train_cutoff, :]
            self.train_y = self.targets[:, :self.train_cutoff, :]

            self.val_X = self.features[:, self.train_cutoff:self.val_cutoff, :]
            self.val_y = self.targets[:, self.train_cutoff:self.val_cutoff, :]

            self.test_X = self.features[:, self.val_cutoff:, :]
            self.test_y = self.targets[:, self.val_cutoff:, :]


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
             + np.random.normal(loc=0.0, scale=noise_level*self.amplitude, size=n.shape[0]) + self.amplitude
            self.sinusoids.append(x.reshape(-1, 1))

        self.sinusoids = np.concatenate(self.sinusoids, axis=1)


    def plot(self):
        for i in range(self.sinusoids.shape[1]):
            plt.plot(self.sinusoids[:, i])
        plt.show()


def MASE(actual_ts, predicted_ts):
    forecast_error = np.mean(np.abs(actual_ts - predicted_ts))
    naive_forecast_error = np.mean(np.abs(actual_ts[:-1] - actual_ts[1:]))
    return forecast_error/naive_forecast_error

def MAPE():
    pass


def main():
    nb_steps_in = 150
    nb_steps_out = 34
    target_index = None
    train_size = 0.7
    val_size = 0.15
    target_index = None
    synthetic_sinusoids = SyntheticSinusoids(num_sinusoids=5,
                                             amplitude=1,
                                             sampling_rate=5000,
                                             length=10000)

    sinusoids = pd.DataFrame(synthetic_sinusoids.sinusoids)
    wavenet_sinusoid_ts = ForecastTimeSeries(sinusoids,
                                             vector_output_mode=False,
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
    lstm_sinusoid_ts = ForecastTimeSeries(sinusoids,
                                          vector_output_mode=True,
                                          train_size=train_size,
                                          val_size=val_size,
                                          nb_steps_in=100,
                                          nb_steps_out=5,
                                          target_index=None)
    index = np.arange(0, 100)
    sns.lineplot(index, lstm_sinusoid_ts.features[0, :, 0], label='predicted')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()







