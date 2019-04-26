import pandas as pd
import matplotlib.pyplot as plt
from multi_tsf.time_series_utils import generate_stats

if __name__ == '__main__':
    path = './wavenet_unconditional/results_df.csv'
    df = pd.read_csv(path, index_col = ['first', 'second']).T
    col_names = df.columns.levels[0].tolist()
    stats_dict = {}
    for col in col_names:
        print(col)
        y_pred = df[(col, 'prediction')].values.reshape(-1, 1)
        y_actual = df[(col, 'actual')].values.reshape(-1, 1)
        mape, mase, rmse = generate_stats(y_actual, y_pred)
        if mape > 3 or mase > 2 or rmse > 100:
            continue
        else:
            stats_dict[col] = [mape*100, mase, rmse]
            print(mape, mase, rmse)
    stats_df = pd.DataFrame.from_dict(stats_dict)
    stats_df.index = ['MAPE', 'MASE', 'RMSE']
    stats_df.loc['RMSE'].plot.barh()
    plt.show()