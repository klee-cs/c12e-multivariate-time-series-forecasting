import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import collections
from multi_tsf.time_series_utils import SyntheticSinusoids, train_val_test_split
from multi_tsf.db_reader import Jackson_GGN_DB
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import sts
from pprint import pprint

def plot_components(dates,
                    component_means_dict,
                    component_stddevs_dict,
                    x_locator=None,
                    x_formatter=None):
  """Plot the contributions of posterior components in a single figure."""
  colors = sns.color_palette()
  c1, c2 = colors[0], colors[1]

  axes_dict = collections.OrderedDict()
  num_components = len(component_means_dict)
  fig = plt.figure(figsize=(12, 2.5 * num_components))
  for i, component_name in enumerate(component_means_dict.keys()):
    component_mean = component_means_dict[component_name]
    component_stddev = component_stddevs_dict[component_name]

    ax = fig.add_subplot(num_components,1,1+i)
    ax.plot(dates, component_mean, lw=2)
    ax.fill_between(dates,
                     component_mean-2*component_stddev,
                     component_mean+2*component_stddev,
                     color=c2, alpha=0.5)
    ax.set_title(component_name)
    if x_locator is not None:
      ax.xaxis.set_major_locator(x_locator)
      ax.xaxis.set_major_formatter(x_formatter)
    axes_dict[component_name] = ax
  fig.autofmt_xdate()
  fig.tight_layout()
  return fig, axes_dict

def plot_one_step_predictive(dates,
                             observed_time_series,
                             one_step_mean,
                             one_step_scale,
                             x_locator=None,
                             x_formatter=None):
  """Plot a time series against a model's one-step predictions."""

  colors = sns.color_palette()
  c1, c2 = colors[0], colors[1]

  fig=plt.figure(figsize=(12, 6))
  ax = fig.add_subplot(1,1,1)
  num_timesteps = one_step_mean.shape[-1]
  ax.plot(dates, observed_time_series, label="observed time series", color=c1)
  ax.plot(dates, one_step_mean, label="one-step prediction", color=c2)
  ax.fill_between(dates,
                  one_step_mean - one_step_scale,
                  one_step_mean + one_step_scale,
                  alpha=0.1, color=c2)
  ax.legend()

  if x_locator is not None:
    ax.xaxis.set_major_locator(x_locator)
    ax.xaxis.set_major_formatter(x_formatter)
    fig.autofmt_xdate()
  fig.tight_layout()
  return fig, ax



def build_model(observed_time_series):
  # trend = sts.LocalLinearTrend(observed_time_series=observed_time_series)
  seasonal_daily = tfp.sts.Seasonal(
      num_seasons=48,
      num_steps_per_season=1,
      observed_time_series=observed_time_series,
      name='day_of_week_effect'
  )
  seasonal_weekly = tfp.sts.Seasonal(
      num_seasons=5,
      num_steps_per_season=48*5,
      observed_time_series=observed_time_series,
      name='week_of_year_effect'
  )
  model = sts.Sum([seasonal_daily, seasonal_weekly], observed_time_series=observed_time_series)

  return model



def main():
    pass

if __name__ == '__main__':
    path = './STS'
    num_forecast_steps = 48*31

    train_df, val_df, test_df = train_val_test_split(data_path='./data/top_volume_active_work_sets.csv',
                                                     save_path='./data',
                                                     test_cutoff_date='2019-01-01',
                                                     train_size=0.8)

    data = train_df.iloc[:, -1].values
    training_data = data[:-num_forecast_steps]

    tf.reset_default_graph()
    model = build_model(observed_time_series=training_data)
    pprint(model.parameters)

    with tf.variable_scope('sts_elbo', reuse=tf.AUTO_REUSE):
        elbo_loss, variational_posteriors = tfp.sts.build_factored_variational_loss(
            model,
            observed_time_series=training_data)

    num_variational_steps = 26  # @param { isTemplate: true}
    num_variational_steps = int(num_variational_steps)
    train_vi = tf.train.AdamOptimizer(0.1).minimize(elbo_loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(path + '/logs/train', sess.graph)
        for i in range(num_variational_steps):
            _, elbo_ = sess.run([train_vi, elbo_loss])
            print("step {} -ELBO {}".format(i, elbo_))
        # Draw samples from the variational posterior.
        q_samples_post_ = sess.run({k: q.sample(10)
                                   for k, q in variational_posteriors.items()})




    forecast_dist = tfp.sts.forecast(
        model,
        observed_time_series=training_data,
        parameter_samples=q_samples_post_,
        num_steps_forecast=num_forecast_steps)

    num_samples = 10

    with tf.Session() as sess:
        forecast_mean, forecast_scale, forecast_samples = sess.run(
            (forecast_dist.mean()[..., 0],
             forecast_dist.stddev()[..., 0],
             forecast_dist.sample(num_samples)[..., 0]))


    forecast_steps = np.arange(0, num_forecast_steps)
    plt.plot(forecast_steps, forecast_samples.T, lw=1, alpha=0.1)
    plt.plot(forecast_steps, data[-num_forecast_steps:], label='actual')
    plt.plot(forecast_steps, forecast_mean, lw=2, ls='--', label='forecast_mean')
    plt.fill_between(forecast_steps,
                    forecast_mean - 2 * forecast_scale,
                    forecast_mean + 2 * forecast_scale, alpha=0.2)
    plt.show()

