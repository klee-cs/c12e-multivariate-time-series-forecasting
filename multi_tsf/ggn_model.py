import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from multi_tsf.WaveNetForecastingModel import WaveNetForecastingModel
from multi_tsf.time_series_utils import ForecastTimeSeries
from multi_tsf.db_reader import Jackson_GGN_DB


def main():
    epochs = 150
    train_size = 0.7
    val_size = 0.15
    batch_size = 64
    nb_dilation_factors = [1, 2, 4, 8, 16, 32, 64, 128]
    nb_layers = len(nb_dilation_factors)
    nb_steps_in = 520
    nb_steps_out = 34
    nb_filters = 25
    target_index = 0
    num_top_skills = 1

    jackson_ggn_db = Jackson_GGN_DB(cache_path='./data')
    skill_ts = jackson_ggn_db.get_summed_work_items_by_skill(start_time='2017-01-31',
                                                             end_time='2019-01-31',
                                                             from_cache=True)

    most_common_skills = skill_ts.sum(axis=0).sort_values(ascending=False)
    skill_ts = skill_ts[most_common_skills.index.tolist()[0:num_top_skills]]
    forecast_data = ForecastTimeSeries(skill_ts,
                                       vector_output_mode=False,
                                       train_size=train_size,
                                       val_size=val_size,
                                       nb_steps_in=nb_steps_in,
                                       nb_steps_out=nb_steps_out,
                                       target_index=target_index)


    period_dates, period_features, period_targets = forecast_data.create_predict_periods(nb_steps_in=nb_steps_in,
                                                                                         nb_steps_out=nb_steps_out,
                                                                                         target_index=target_index,
                                                                                         predict_hour=7)


    ##################WaveNet######################
    wavenet = WaveNetForecastingModel(name='WaveNet',
                                      nb_layers=nb_layers,
                                      nb_filters=nb_filters,
                                      nb_dilation_factors=nb_dilation_factors,
                                      nb_input_features=forecast_data.nb_input_features,
                                      nb_output_features=forecast_data.nb_output_features)

    wavenet.fit(forecast_data=forecast_data,
                model_path='./wavenet_test',
                epochs=epochs,
                batch_size=batch_size)

    predictions = wavenet.predict(period_features, period_targets, nb_steps_out=nb_steps_out)



    jackson_ggn_db.close()

if __name__ == '__main__':
    main()