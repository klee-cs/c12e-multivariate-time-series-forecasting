import pandas as pd
import matplotlib.pyplot as plt
from multi_tsf.WaveNetForecastingModel import WaveNetForecastingModel
from multi_tsf.time_series_utils import ForecastTimeSeries
from multi_tsf.db_reader import Jackson_GGN_DB


def main():
    epochs = 1000
    batch_size = 1
    conditional = False
    nb_dilation_factors = [1, 2, 4, 8, 16, 32, 64]
    nb_layers = len(nb_dilation_factors)
    nb_filters = 16
    nb_steps_in = 336
    nb_steps_out = 34
    predict_hour = 6

    jackson_ggn_db = Jackson_GGN_DB(cache_path='./data')
    skill_ts = jackson_ggn_db.get_summed_work_items_by_skill(start_date='2017-01-31',
                                                             end_date='2019-03-22',
                                                             start_hour=0,
                                                             end_hour=24,
                                                             include_weekend=False,
                                                             use_default_skills=True,
                                                             from_cache=True)


    # skill_ts = pd.DataFrame(skill_ts.iloc[:, 0:5])
    nb_input_features = skill_ts.shape[1]
    forecast_data = ForecastTimeSeries(skill_ts,
                                       vector_output_mode=False,
                                       test_cutoff_date='2019-01-01',
                                       nb_steps_in=nb_steps_in,
                                       nb_steps_out=nb_steps_out,
                                       predict_hour=predict_hour)


    wavenet = WaveNetForecastingModel(name='WaveNet',
                                      conditional=conditional,
                                      nb_layers=nb_layers,
                                      nb_filters=nb_filters,
                                      nb_dilation_factors=nb_dilation_factors,
                                      nb_input_features=nb_input_features,
                                      batch_size=batch_size,
                                      lr=1e-3,
                                      model_path='./wavenet_test',
                                      ts_names = skill_ts.columns.to_list())

    wavenet.fit(forecast_data, epochs=epochs)

    jackson_ggn_db.close()


if __name__ == '__main__':
    main()