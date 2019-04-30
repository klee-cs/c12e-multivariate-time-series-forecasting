from multi_tsf.db_reader import Jackson_GGN_DB
from multi_tsf.WaveNetForecastingModel import WaveNetForecastingModel
from multi_tsf.time_series_utils import ForecastTimeSeries
import pandas as pd

if __name__ == '__main__':
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

    skill_ts = pd.DataFrame(skill_ts.iloc[:, 0:3])
    forecast_data = ForecastTimeSeries(skill_ts,
                                       vector_output_mode=False,
                                       test_cutoff_date='2019-01-01',
                                       nb_steps_in=nb_steps_in,
                                       nb_steps_out=nb_steps_out,
                                       predict_hour=predict_hour)
    wavenet_forecasting_model = WaveNetForecastingModel.restore_model(model_params_path='./wavenet_conditional/model_params.json')
    wavenet_forecasting_model.evaluate(forecast_data, set='Validation')