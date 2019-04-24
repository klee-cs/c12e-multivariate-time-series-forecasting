import psycopg2
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import List, Tuple
from datetime import datetime

class Jackson_GGN_DB(object):

    def __init__(self, cache_path: str) -> None:
        self.default_skills = ['pp-clip',
                               'pp-blank',
                               'fullpageentry',
                               'inforce',
                               'nbforms',
                               'clip',
                               'marginnote',
                               'POS Annuity Disbursement File Review Manual',
                               'POS Disbursement Validation Manual',
                               'bmfe',
                               'BM PRODUCER OF RECORD CHANGES',
                               'LOI Send to File QC Manual',
                               'POS Beneficiary Change',
                               'inforceclaims',
                               'POS Annuity Disbursement Process Manual',
                               'OPS Claims Annuity',
                               'POS Follow Up Required',
                               'POS Simple Resolution',
                               'POS Overnight Prep',
                               'POS Ownership Change',
                               'POS TP Auth - VA VUL',
                               'POS Trade Work around',
                               'BM Electronic Funds Transfer']
        try:
            self.cache_path = cache_path
            os.makedirs(self.cache_path, exist_ok=True)
            self.connection = psycopg2.connect(user="jackson_ml",
                                               password="R284o78WgsYw",
                                               host="jackson-dev.cl0yhf6smpns.us-east-2.rds.amazonaws.com",
                                               port="5432",
                                               database="jackson_ggn")
            self.cursor = self.connection.cursor()
            # Print PostgreSQL Connection properties
            print(self.connection.get_dsn_parameters(), "\n")
            # Print PostgreSQL version
            self.cursor.execute("SELECT version();")
            record = self.cursor.fetchone()
            print("You are connected to - ", record, "\n")

        except (Exception, psycopg2.Error) as error:
            print("Error while connecting to PostgreSQL", error)

    def close(self) -> None:
        self.cursor.close()

    def get_all_skills(self) -> pd.DataFrame:
        query = "select distinct(skill_display_nm) from work_skill_log;"
        df = pd.read_sql(query, self.connection)
        return df

    def get_all_work_sets(self) -> pd.DataFrame:
        query = "select distinct(work_set_id) from work_set_log;"
        df = pd.read_sql(query, self.connection)
        return df

    def get_summed_work_items_by_work_set(self,
                                          start_date: str,
                                          end_date: str,
                                          start_hour: int,
                                          end_hour: int,
                                          include_weekend: bool=False,
                                          from_cache=False) -> pd.DataFrame:
        if from_cache:
            df = pd.read_csv(self.cache_path + '/summed_work_items_by_work_set.csv', index_col=0)
            df = df.set_index('received_ts_rounded')
            df.index = pd.to_datetime(df.index)
            return df
        else:
            query = "SELECT wil_cs.received_ts_rounded, wil_cs.work_set_id, skill_nm.skill_display_nm, wil_cs.rec_item_count \
            FROM (select received_ts_rounded, work_set_id, count(work_item_id) as rec_item_count \
            from work_item_log_cs \
            where received_ts_rounded >= '%s'\
            and received_ts_rounded <= '%s'\
            group by received_ts_rounded, work_set_id) as wil_cs \
            LEFT JOIN \
            (select distinct on (skill_display_nm, work_set_id) skill_display_nm, work_set_id \
            from work_skill_log a JOIN work_set_log b \
            on a.work_skill_id=b.work_skill_id) as skill_nm \
            on skill_nm.work_set_id = wil_cs.work_set_id; " % (start_date, end_date)
            df = pd.read_sql(query, self.connection)
            df.dropna(inplace=True)
            df = df.pivot(index='received_ts_rounded', columns='skill_display_nm', values='rec_item_count')
            df_timestamps = pd.DataFrame(pd.date_range(start_date, end_date, freq='30T'), columns=['received_ts_rounded'])
            df = df_timestamps.join(df, how='left', on='received_ts_rounded')
            df = df.set_index('received_ts_rounded')
            del df_timestamps
            df = df.fillna(value=0)
            df = self.extract_subset_data(df, start_hour, end_hour, include_weekend=include_weekend)
            df.to_csv(self.cache_path + '/summed_work_items_by_work_set.csv')
            return df


    def get_summed_work_items_by_skill(self,
                                       start_date: str,
                                       end_date: str,
                                       start_hour: int,
                                       end_hour: int,
                                       include_weekend: bool =False,
                                       use_default_skills: bool = False,
                                       from_cache=False) -> pd.DataFrame:
        if from_cache:
            df = pd.read_csv(self.cache_path + '/summed_work_items_by_skill.csv', index_col=0)
            df.index = pd.to_datetime(df.index)
            return df
        else:
            query = "select wil_cs.received_ts_rounded, skill_nm.skill_display_nm, sum(wil_cs.rec_item_count) as rec_item_count from \
            (select received_ts_rounded, work_set_id, count(work_item_id) as rec_item_count \
            from work_item_log_cs \
            where received_ts_rounded >= '%s'\
            and received_ts_rounded <= '%s'\
            group by received_ts_rounded, work_set_id) as wil_cs\
            LEFT JOIN\
            (select distinct on (skill_display_nm, work_set_id) skill_display_nm, work_set_id\
            from work_skill_log a JOIN work_set_log b \
            on a.work_skill_id=b.work_skill_id) as skill_nm \
            on skill_nm.work_set_id = wil_cs.work_set_id\
            group by wil_cs.received_ts_rounded, skill_nm.skill_display_nm;" % (start_date, end_date)
            df = pd.read_sql(query, self.connection)
            df.dropna(inplace=True)
            df = df.pivot(index='received_ts_rounded', columns='skill_display_nm', values='rec_item_count')
            df_timestamps = pd.DataFrame(pd.date_range(start_date, end_date, freq='30T'), columns=['received_ts_rounded'])
            df = df_timestamps.join(df, how='left', on='received_ts_rounded')
            df = df.set_index('received_ts_rounded')
            del df_timestamps
            df = df.fillna(value=0)
            df = self.extract_subset_data(df, start_hour, end_hour, include_weekend=include_weekend)
            if use_default_skills:
                print(self.default_skills)
                df = df[self.default_skills]
            df.to_csv(self.cache_path + '/summed_work_items_by_skill.csv')
            return df

    @classmethod
    def extract_subset_data(self,
                            df: pd.DataFrame,
                            start_hour: int=6,
                            end_hour: int=23,
                            include_weekend: bool=False):
        '''
        Inputs:
        ----------
        df:
            The dataframe to be transformed
        start_hour:
            Start time of the day. Default : 6 am
        end_hour:
            Default : 11 pm
        include_weekend:
            Indicator for weekends. False is excluded and True otherwise

        Output:
        ------------
        Dataframe which is a subset of the input provided

        '''
        # Narrowing down the hours of the day
        df = df[(df.index.hour >= start_hour) & (df.index.hour <= end_hour)]
        if not include_weekend:
            # excluding weekends
            df = df[df.index.dayofweek.isin([0, 1, 2, 3, 4])]  # Friday:4 , Monday: 0
        return df

    @classmethod
    def find_active_date_ranges(self, master_df: pd.DataFrame) -> List[Tuple]:
        active_date_ranges = []
        start_idx = None
        start_date = None
        end_idx = None
        end_date = None
        for col in range(master_df.shape[1]):
            # find start time
            for idx, value in enumerate(master_df.iloc[:, col]):
                if not (value is None or value == 0 or np.isnan(value)):
                    start_idx = idx
                    start_date = master_df.iloc[:, col].index[start_idx]
                    break
            # find end time
            for idx, value in enumerate(master_df.iloc[::-1, col]):
                if not (value is None or value == 0 or np.isnan(value)):
                    end_idx = master_df.shape[0] - idx - 1
                    end_date = master_df.iloc[:, col].index[end_idx]
                    break
            active_date_ranges.append((start_idx, start_date, end_idx, end_date))
        return active_date_ranges

    @classmethod
    def rank_sparsity(self, master_df: pd.DataFrame, active_date_ranges: List[Tuple]) -> pd.DataFrame:
        sparsities = []
        num_skills = master_df.shape[1]
        for col in range(num_skills):
            ts = master_df.iloc[active_date_ranges[col][0]:active_date_ranges[col][2], col]
            num_rows = ts.shape[0]
            sparsity = (np.sum(ts == 0) + np.sum(ts is None) + np.sum(np.isnan(ts))) / num_rows
            sparsities.append(sparsity)
        skill_names = master_df.columns.tolist()
        ranked_sparsity = pd.DataFrame({'skill_name': skill_names, 'sparsity': sparsities})
        return ranked_sparsity.sort_values('sparsity')

    @classmethod
    def plot_skill_ts(ts, skill_name):
        start_date = ts.index[0]
        end_date = ts.index[-1]
        date_range = pd.DataFrame(pd.date_range(start_date, end_date, freq='30T'), columns=['timestamps'])
        weekends = date_range[(date_range.timestamps.dt.dayofweek.isin([5, 6]))]
        offtimes = date_range[date_range.timestamps.dt.dayofweek.isin([0, 1, 2, 3, 4]) &
                              ((date_range.timestamps.dt.time < datetime.time(6, 0)) |
                               (date_range.timestamps.dt.time > datetime.time(23, 0)))]
        weekends_and_offtimes = pd.concat([weekends, offtimes])
        weekends_and_offtimes = weekends_and_offtimes.set_index('timestamps')
        weekends_and_offtimes = weekends_and_offtimes.set_index(pd.to_datetime(weekends_and_offtimes.index))
        plt.close()
        plt.scatter(weekends_and_offtimes.index, [0] * len(weekends_and_offtimes))
        plt.scatter(ts.index, ts)
        plt.title('Time Series for Skill: ' + skill_name)
        plt.xlabel('Time')
        plt.ylabel('Item Count')
        plt.show()


if __name__ == '__main__':
    jackson_ggn_db = Jackson_GGN_DB(cache_path='./data')
    skill_ts = jackson_ggn_db.get_summed_work_items_by_skill(start_date='2011-12-28',
                                                             end_date='2019-03-23',
                                                             start_hour=6,
                                                             end_hour=11,
                                                             include_weekend=False,
                                                             use_default_skills=True,
                                                             from_cache=False)
    date_ranges = Jackson_GGN_DB.find_active_date_ranges(skill_ts)
    skill_ts.plot()
    plt.show()
    jackson_ggn_db.close()