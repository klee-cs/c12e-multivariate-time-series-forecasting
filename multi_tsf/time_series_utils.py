import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import scipy as sp
import tensorflow as tf

class ForecastTimeSeries(object):

    def __init__(self,
                 ts_df: pd.DataFrame,
                 vector_output_mode: bool,
                 test_cutoff_date: str,
                 nb_steps_in: int,
                 nb_steps_out: int,
                 predict_hour: int) -> None:
        self.ts_df = ts_df
        self.vector_output_mode = vector_output_mode
        self.test_cutoff_date = test_cutoff_date
        self.column_names = ts_df.columns
        self.time_series = ts_df.values
        self.dates = ts_df.index.tolist()
        self.nb_steps_out = nb_steps_out
        self.nb_steps_in = nb_steps_in
        self.predict_hour = predict_hour


        self.train_df, self.val_df, self.test_df = self._split_train_validation_test(self.ts_df, self.test_cutoff_date)


        self.nb_input_features = ts_df.shape[-1]
        self.nb_output_features = 1
        self.reshaped_rolling = {'Train':
                                      {
                                          'features': None,
                                          'targets': None
                                      },
                                  'Validation':
                                      {
                                          'features': None,
                                          'targets': None
                                      },
                                  'Test':
                                      {
                                          'features': None,
                                          'targets': None
                                      }
                                }
        self.reshaped_periods = {'Train':
                                      {
                                          'features': None,
                                          'targets': None,
                                          'dates': None
                                      },
                                      'Validation':
                                      {
                                          'features': None,
                                          'targets': None,
                                          'dates': None
                                      },
                                      'Test':
                                      {
                                          'features': None,
                                          'targets': None,
                                          'dates': None
                                      }
                                }


        if self.vector_output_mode:
            self._create_lags_and_target()
        else:
            self.reshaped_rolling['Train']['features'], \
            self.reshaped_rolling['Train']['targets'] = self._create_shifted_feature_targets(self.train_df)

            self.reshaped_rolling['Validation']['features'], \
            self.reshaped_rolling['Validation']['targets'] = self._create_shifted_feature_targets(self.val_df)

            self.reshaped_rolling['Test']['features'], \
            self.reshaped_rolling['Test']['targets'] = self._create_shifted_feature_targets(self.test_df)

            self.reshaped_periods['Train']['features'], \
            self.reshaped_periods['Train']['targets'], \
            self.reshaped_periods['Train']['dates'] = self._create_predict_periods(self.train_df,
                                                                                   nb_steps_out,
                                                                                   nb_steps_in,
                                                                                   predict_hour)
            self.reshaped_periods['Validation']['features'], \
            self.reshaped_periods['Validation']['targets'], \
            self.reshaped_periods['Validation']['dates'] = self._create_predict_periods(self.val_df,
                                                                                        nb_steps_out,
                                                                                        nb_steps_in,
                                                                                        predict_hour)
            self.reshaped_periods['Test']['features'], \
            self.reshaped_periods['Test']['targets'], \
            self.reshaped_periods['Test']['dates'] = self._create_predict_periods(self.test_df,
                                                                                  nb_steps_out,
                                                                                  nb_steps_in,
                                                                                  predict_hour)



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

            seq_x, seq_y = self.time_series[i:end_ix, :], self.time_series[end_ix:out_end_ix, :]
            X.append(seq_x)
            y.append(seq_y)

        self.features = np.array(X)
        self.targets = np.array(y)
        self.nb_input_features = self.features.shape[-1]
        self.nb_output_features = self.features.shape[-1]


    def _create_shifted_feature_targets(self,
                                        ts_df: pd.DataFrame) -> (np.array, np.array):
        time_series = ts_df.values

        seq_x, seq_y = time_series[:-1, :], time_series[1:, :]

        #Expand for batch dimension = 1
        features = np.expand_dims(seq_x, axis=0)
        targets = np.expand_dims(seq_y, axis=0)
        return features, targets


    def _create_predict_periods(self,
                                ts_df: pd.DataFrame,
                                nb_steps_out: int,
                                nb_steps_in: int,
                                predict_hour: int) -> (np.array, np.array):

        time_series = ts_df.values
        date_times = ts_df.index.to_list()
        features = []
        targets = []
        dts = []
        for i in np.arange(0, ts_df.shape[0]):
            hour = date_times[i].hour
            min = date_times[i].minute
            if hour == predict_hour and min == 0:
                if (i - nb_steps_in) >= 0 and (i + nb_steps_out) < ts_df.shape[0]:
                    feature = time_series[i - nb_steps_in:i, :]
                    target = time_series[i:i + nb_steps_out, :]
                    features.append(feature)
                    targets.append(target)
                    dts += date_times[i:i + nb_steps_out]

        features = np.array(features)
        targets = np.array(targets)
        return features, targets, dts



    def _split_train_validation_test(self,
                                     ts_df: pd.DataFrame,
                                     test_cutoff_date: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        train_size = 0.8
        if test_cutoff_date:
            ts_df = ts_df.sort_index()
            test_cutoff_idx = ts_df.index.get_loc(test_cutoff_date).start
        else:
            test_cutoff_idx = math.ceil(ts_df.shape[0] * train_size)
        val_cutoff_idx = math.ceil(test_cutoff_idx * train_size)

        self.train_df = ts_df.iloc[:test_cutoff_idx, :]
        self.test_df = ts_df.iloc[test_cutoff_idx:, :]
        self.val_df = self.train_df.iloc[val_cutoff_idx:, :]
        self.train_df = ts_df.iloc[:val_cutoff_idx, :]

        return self.train_df, self.val_df, self.test_df


class SyntheticSinusoids(object):

    def __init__(self,
                 num_sinusoids: int,
                 amplitude: float,
                 sampling_rate: int,
                 length: int,
                 frequency: float = None):
        self.num_sinusoids = num_sinusoids
        self.amplitude = amplitude
        self.sampling_rate = sampling_rate
        self.length = length
        n = np.arange(0, self.length)

        if frequency is  None:
            self.sinusoids = []
            for i in range(num_sinusoids):
                f = np.random.uniform(int(sampling_rate//100), int(self.sampling_rate//30))
                noise_level = np.random.uniform(0.01, 0.02)
                x = self.amplitude*np.cos(2*np.pi*f/self.sampling_rate*n) \
                 + np.random.normal(loc=0.0, scale=noise_level*self.amplitude, size=n.shape[0]) + self.amplitude
                self.sinusoids.append(x.reshape(-1, 1))

            self.sinusoids = np.concatenate(self.sinusoids, axis=1)
        else:
            noise_level = np.random.uniform(0.01, 0.02)
            self.sinusoids = self.amplitude * np.cos(2 * np.pi * frequency / self.sampling_rate * n) \
                + np.random.normal(loc=0.0, scale=noise_level * self.amplitude, size=n.shape[0]) + self.amplitude


    def plot(self):
        for i in range(self.sinusoids.shape[1]):
            plt.plot(self.sinusoids[:, i])
            plt.show()


def generate_stats(trueY, forecastY, missing=True):
    """ From TRMF code """
    nz_mask = trueY != 0
    diff = forecastY - trueY
    abs_true = sp.absolute(trueY)
    abs_diff = sp.absolute(diff)

    def my_mean(x):
        tmp = x[sp.isfinite(x)]
        assert len(tmp) != 0
        return tmp.mean()


    with sp.errstate(divide='ignore'):
        # rmse
        rmse = sp.sqrt((diff ** 2).mean())
        # normalized root mean squared error
        nrmse = sp.sqrt((diff ** 2).mean()) / abs_true.mean()

        # baseline
        abs_baseline = sp.absolute(trueY[1:, :] - trueY[:-1, :])
        mase = abs_diff.mean() / abs_baseline.mean()
        m_mase = my_mean(abs_diff.mean(axis=0) / abs_baseline.mean(axis=0))

        mape = my_mean(sp.divide(abs_diff, abs_true, where=nz_mask))

        return mape, mase, rmse















