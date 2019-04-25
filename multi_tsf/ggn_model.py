import pandas as pd
import matplotlib.pyplot as plt
from multi_tsf.WaveNetForecastingModel import WaveNetForecastingModel, CompositeWaveNetForecastingModel
from multi_tsf.time_series_utils import ForecastTimeSeries
from multi_tsf.db_reader import Jackson_GGN_DB


def main():
    epochs = 100
    batch_size = 128
    conditional = False
    nb_dilation_factors = [1, 2, 4, 8, 16, 32, 64]
    nb_layers = len(nb_dilation_factors)
    nb_steps_in = 150
    nb_steps_out = 34
    nb_filters = 16
    target_index = 0
    predict_hour = 6

    jackson_ggn_db = Jackson_GGN_DB(cache_path='./data')
    skill_ts = jackson_ggn_db.get_summed_work_items_by_skill(start_date='2017-01-31',
                                                             end_date='2019-03-22',
                                                             start_hour=0,
                                                             end_hour=24,
                                                             include_weekend=False,
                                                             use_default_skills=True,
                                                             from_cache=True)


    skill_ts = skill_ts.iloc[:,0:5]

    composite_wavenet = CompositeWaveNetForecastingModel(nb_steps_in=nb_steps_in,
                                                         nb_steps_out=nb_steps_out,
                                                         predict_hour=predict_hour,
                                                         conditional=conditional,
                                                         nb_layers=nb_layers,
                                                         nb_filters=nb_filters,
                                                         nb_dilation_factors=nb_dilation_factors)


    composite_wavenet.fit(skill_ts,
                          model_path='./composite_wavenet',
                          epochs=epochs,
                          batch_size=batch_size)



    exit(0)

    forecast_data = ForecastTimeSeries(skill_ts,
                                       vector_output_mode=False,
                                       test_cutoff_date='2019-01-01',
                                       nb_steps_in=nb_steps_in,
                                       nb_steps_out=nb_steps_out,
                                       target_index=target_index,
                                       predict_hour=predict_hour)

    wavenet = WaveNetForecastingModel(name='WaveNet',
                                      conditional=conditional,
                                      nb_layers=nb_layers,
                                      nb_filters=nb_filters,
                                      nb_dilation_factors=nb_dilation_factors,
                                      nb_input_features=forecast_data.nb_input_features,
                                      nb_output_features=forecast_data.nb_output_features)

    wavenet.fit(forecast_data=forecast_data,
                model_path='./wavenet_test',
                epochs=epochs,
                batch_size=batch_size)

    mape, mase, rmse = wavenet.evaluate(forecast_data)
    print(mape)
    print(mase)
    print(rmse)

    predictions = wavenet.predict(forecast_data.reshaped_periods['Test']['features'][-1],
                                  nb_steps_out=nb_steps_out)

    pd.DataFrame(predictions).plot()
    plt.show()


    jackson_ggn_db.close()

if __name__ == '__main__':
    main()