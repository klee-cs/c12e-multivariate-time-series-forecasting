import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from multi_tsf.WaveNetForecastingModel import WaveNetForecastingModel
from multi_tsf.LSTMForecastingModel import LSTMForecastingModel
from multi_tsf.time_series_utils import SyntheticSinusoids, ForecastTimeSeries
from multi_tsf.db_reader import Jackson_GGN_DB

def main():
    epochs = 15
    train_size = 0.7
    val_size = 0.15
    batch_size = 64
    nb_dilation_factors = [1, 2, 4, 8]
    nb_layers = len(nb_dilation_factors)
    nb_filters = 64
    nb_steps_in = 336
    target_index = None
    num_top_skills = 3

    jackson_ggn_db = Jackson_GGN_DB(cache_path='./data')
    skill_ts = jackson_ggn_db.get_summed_work_items_by_skill(start_time='2017-01-31',
                                                             end_time='2019-01-31',
                                                             from_cache=False)

    most_common_skills = skill_ts.sum(axis=0).sort_values(ascending=False)
    skill_ts = skill_ts[most_common_skills.index.tolist()[0:num_top_skills]]

    #################LSTM########################
    # nb_steps_out = 1
    # forecast_data = ForecastTimeSeries(skill_ts.values,
    #                                    train_size=train_size,
    #                                    val_size=val_size,
    #                                    nb_steps_in=nb_steps_in,
    #                                    nb_steps_out=nb_steps_out,
    #                                    target_index=target_index)
    #
    # lstm = LSTMForecastingModel(name='LSTM',
    #                             nb_decoder_layers=2,
    #                             nb_encoder_layers=2,
    #                             nb_units=100,
    #                             nb_steps_in=nb_steps_in,
    #                             nb_steps_out=nb_steps_out,
    #                             nb_input_features=forecast_data.nb_input_features,
    #                             nb_output_features=forecast_data.nb_output_features)
    #
    # lstm.fit(forecast_data=forecast_data,
    #             model_path='./lstm_test',
    #             epochs=epochs,
    #             batch_size=batch_size)
    #
    # predicted_ts, actual_ts = lstm.predict_historical(forecast_data=forecast_data,
    #                                                      set='Validation',
    #                                                      plot=True,
    #                                                      num_features_display=num_top_skills)
    #
    #
    # exit(0)

    ##################WaveNet######################
    nb_steps_out = None
    forecast_data = ForecastTimeSeries(skill_ts.values,
                                       train_size=train_size,
                                       val_size=val_size,
                                       nb_steps_in=nb_steps_in,
                                       nb_steps_out=nb_steps_out,
                                       target_index=target_index)


    wavenet = WaveNetForecastingModel(name='WaveNet',
                                      nb_layers=nb_layers,
                                      nb_filters=nb_filters,
                                      nb_dilation_factors=nb_dilation_factors,
                                      max_look_back=nb_steps_in,
                                      nb_input_features=forecast_data.nb_input_features,
                                      nb_output_features=forecast_data.nb_output_features)

    wavenet.fit(forecast_data=forecast_data,
                model_path='./wavenet_test',
                epochs=epochs,
                batch_size=batch_size)

    predicted_ts, actual_ts = wavenet.predict_historical(forecast_data=forecast_data,
                                                         set='Validation',
                                                         plot=True,
                                                         num_features_display=num_top_skills)

    jackson_ggn_db.close()
if __name__ == '__main__':
    main()