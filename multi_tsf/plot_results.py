import pandas as pd
import matplotlib.pyplot as plt
from multi_tsf.time_series_utils import generate_stats

if __name__ == '__main__':
    path = '/Users/klee/Dropbox (Cognitive Scale)/Jackson/GGN/results_cache/results_df.csv'
    df = pd.read_csv(path, index_col = ['first', 'second']).T
    col_names = df.columns.levels[0].tolist()
    stats_dict = {}
    for col in col_names:
        print(col)
        # df[col].plot()
        plt.show()
        y_pred = df[(col, 'prediction')].values.reshape(-1, 1)
        y_actual = df[(col, 'actual')].values.reshape(-1, 1)
        y_pred[y_pred > 2000] = 0
        y_pred[y_pred < 0] = 0
        mape, mase, rmse = generate_stats(y_actual, y_pred)
        stats_dict[col] = [mape * 100, mase, rmse]
        print(mape, mase, rmse)
    stats_df = pd.DataFrame.from_dict(stats_dict)
    stats_df.index = ['MAPE', 'MASE', 'RMSE']
    fig, ax = plt.subplots(1, 3, sharey='row')
    stats_df.loc['RMSE'].plot.barh(ax=ax[0], title='RMSE')
    stats_df.loc['MAPE'].plot.barh(ax=ax[1], title='MAPE')
    stats_df.loc['MASE'].plot.barh(ax=ax[2], title='MASE')
    plt.show()