import psycopg2
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import List, Tuple
from datetime import datetime
import pickle

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
                                          work_set_list=None) -> pd.DataFrame:
        if work_set_list is not None:
            work_set_list = ",".join(work_set_list)
            query_filter = ' and work_set_id IN (%s) ' % work_set_list
        else:
            query_filter = ' '
        query = "SELECT wil_cs.received_ts_rounded, wil_cs.work_set_id, wil_cs.rec_item_count \
        FROM (select received_ts_rounded, work_set_id, count(work_item_id) as rec_item_count \
        from work_item_log_cs \
        where received_ts_rounded >= '%s'\
        and received_ts_rounded <= '%s'" % (start_date, end_date) \
                + query_filter \
                + "group by received_ts_rounded, work_set_id) as wil_cs;"
        df = pd.read_sql(query, self.connection)
        df.dropna(inplace=True)
        df = df.pivot(index='received_ts_rounded', columns='work_set_id', values='rec_item_count')
        df_timestamps = pd.DataFrame(pd.date_range(start_date, end_date, freq='30T'), columns=['received_ts_rounded'])
        df = df_timestamps.join(df, how='left', on='received_ts_rounded')
        df = df.set_index('received_ts_rounded')
        del df_timestamps
        df = df.fillna(value=0)
        df = self.extract_subset_data(df, start_hour, end_hour, include_weekend=include_weekend)
        return df


    def get_summed_work_items_by_skill(self,
                                       start_date: str,
                                       end_date: str,
                                       start_hour: int,
                                       end_hour: int,
                                       include_weekend: bool =False,
                                       skill_list=None) -> pd.DataFrame:
        if skill_list is not None:
            skill_list = map(lambda x: "\'" + x + "\'", skill_list)
            skill_list = ",".join(skill_list)
            query_filter = ' WHERE skill_display_nm IN (%s) ' % skill_list
        else:
            query_filter = ''
        query = "select wil_cs.received_ts_rounded, skill_nm.skill_display_nm, sum(wil_cs.rec_item_count) as rec_item_count from \
        (select received_ts_rounded, work_set_id, count(work_item_id) as rec_item_count \
        from work_item_log_cs \
        where received_ts_rounded >= '%s' \
        and received_ts_rounded <= '%s' \
        group by received_ts_rounded, work_set_id) as wil_cs \
        LEFT JOIN \
        (select distinct on (skill_display_nm, work_set_id) skill_display_nm, work_set_id\
        from work_skill_log a JOIN work_set_log b on a.work_skill_id=b.work_skill_id" % (start_date, end_date) \
                + query_filter \
                + ") as skill_nm on skill_nm.work_set_id = wil_cs.work_set_id \
                group by wil_cs.received_ts_rounded, skill_nm.skill_display_nm;"
        df = pd.read_sql(query, self.connection)
        df.dropna(inplace=True)
        df = df.pivot(index='received_ts_rounded', columns='skill_display_nm', values='rec_item_count')
        df_timestamps = pd.DataFrame(pd.date_range(start_date, end_date, freq='30T'), columns=['received_ts_rounded'])
        df = df_timestamps.join(df, how='left', on='received_ts_rounded')
        df = df.set_index('received_ts_rounded')
        del df_timestamps
        df = df.fillna(value=0)
        df = self.extract_subset_data(df, start_hour, end_hour, include_weekend=include_weekend)
        return df


    def filter_active_work_sets(self,
                                start_date,
                                end_date,
                                active_cutoff,
                                percent_cutoff,
                                from_cache=False):
        '''

        :param start_date:
        :param end_date:
        :param active_cutoff:
        :param percent_cutoff:
        :return:
        '''
        if from_cache:
            master_df = pd.read_csv(self.cache_path + '/top_volume_active_work_sets.csv', index_col=0)
            master_df.index = pd.to_datetime(master_df.index)
            return master_df
        else:
            pivot_df = self.get_summed_work_items_by_work_set(start_date=start_date,
                                                              end_date=end_date,
                                                              start_hour=6,
                                                              end_hour=23,
                                                              include_weekend=False,
                                                              work_set_list=None)
            active_date_ranges = self.find_active_date_ranges(pivot_df)
            active_work_sets = []
            for adr in active_date_ranges:
                if adr['end_date'] >= pd.Timestamp(active_cutoff):
                    active_work_sets.append(adr['name'])
            top_volume_active_work_sets = self.top_n(start_date,
                                                     end_date,
                                                     work_set_level=True,
                                                     work_set_list=active_work_sets,
                                                     percent_=percent_cutoff)
            pickle.dump(top_volume_active_work_sets, open(self.cache_path + '/top_volume_work_sets.p', 'wb+'))
            master_df = pivot_df[top_volume_active_work_sets]
            master_df.to_csv(self.cache_path + '/top_volume_active_work_sets.csv')
            return master_df



    def plot_pareto(self, top_n):
        pareto_df = self.pareto_df
        pareto_df['work_set_id'] = pareto_df.work_set_id.astype('str')
        plt.rcParams["figure.figsize"] = (30, 7)
        fig, ax = plt.subplots()
        ax.bar(pareto_df.head(top_n).work_set_id, pareto_df.head(top_n)["percent_vol"], color="C0", width=0.1)
        plt.xticks(rotation=90, fontsize=15)
        plt.ylabel('Percentage total volume', fontsize=20)
        plt.title('Top ' + str(top_n) + ' work sets', fontsize=25)
        ax2 = ax.twinx()
        ax2.plot(pareto_df.head(top_n).work_set_id, pareto_df.head(top_n)["cumpercentage"], color="r", marker="D", ms=7)
        # ax2.yaxis.set_major_formatter(plt.PercentFormatter())
        plt.savefig('Pareto_work_sets.pdf')


    def top_n(self,
              start_date,
              end_date,
              work_set_level=False,
              work_set_list=None,
              percent_=90):
        """
        work_set_level: Default is False and returns work_skill level. if set to True, returns work_set level
        percent_ : cumulative % of work_item count we are interested in
        """

        if work_set_level:

            pareto_df = pd.read_sql(
                "select work_set_id, count(work_item_id) as rec_item_count from work_item_log_cs \
                where received_ts_rounded >= '%s' \
                and received_ts_rounded <= '%s' \
                group by work_set_id;" % (start_date, end_date),
                self.connection)
            if work_set_list is not None:
                pareto_df = pareto_df[pareto_df['work_set_id'].isin(work_set_list)]
            pareto_df = pareto_df.sort_values(by='rec_item_count', ascending=False)
            # pareto_df['work_set_id'] = pareto_df.work_set_id.astype('str')
            pareto_df['percent_vol'] = pareto_df['rec_item_count'] * 100 / pareto_df['rec_item_count'].sum()
            pareto_df["cumpercentage"] = pareto_df["percent_vol"].cumsum()
            self.pareto_df = pareto_df
            self.pareto_df.to_csv(self.cache_path + '/pareto_df.csv')
            return pareto_df[pareto_df.cumpercentage < percent_]['work_set_id'].to_list()
        else:

            pareto_df = pd.read_sql("select skill_nm.skill_display_nm, sum(wil_cs.rec_item_count) as rec_item_count from \
                    (select received_ts_rounded, work_set_id, count(work_item_id) as rec_item_count \
                    from work_item_log_cs \
                    where received_ts_rounded >= '%s' \
                    and received_ts_rounded <= '%s' \
                    group by received_ts_rounded, work_set_id) as wil_cs \
                    LEFT JOIN\
                    (select distinct on (skill_display_nm, work_set_id) skill_display_nm, work_set_id \
                    from work_skill_log a JOIN work_set_log b \
                    on a.work_skill_id=b.work_skill_id) as skill_nm \
                    on skill_nm.work_set_id = wil_cs.work_set_id\
                    group by skill_nm.skill_display_nm;" % (start_date, end_date), self.connection)
            pareto_df['Percent_vol'] = pareto_df['rec_item_count'] * 100 / pareto_df['rec_item_count'].sum()
            pareto_df["cumpercentage"] = pareto_df["percent_vol"].cumsum()
            return pareto_df[pareto_df.cumpercentage < percent_]['skill_display_nm'].to_list()

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
    def find_active_date_ranges(cls, df: pd.DataFrame) -> List[dict]:
        """
        :param df: DataFrame returned by get_summed_work_items_by_skill() or get_summed_work_items_by_work_set()
        :return: start and end indices and dates, and length of activity for each skill or work item in input DataFrame.
        If skill or work item has no non-zero entries, returns None for active_length.
        """
        active_date_ranges = []
        names = df.columns.tolist()

        # loop through each skill or work set
        for col in range(df.shape[1]):
            d = {'name': names[col],
                 'idx': col}
            ts = df.iloc[:, col]
            ts_non_zero = ts.loc[ts > 0]

            # find start date
            if ts_non_zero.empty:
                d['start_date'] = pd.Timestamp('1900-01-01')
            else:
                d['start_date'] = ts_non_zero.index[0]

            # find end date
            if ts_non_zero.empty:
                d['end_date'] = pd.Timestamp('1900-01-01')
            else:
                d['end_date'] = ts_non_zero.index[-1]

            # find active length
            if d['end_date'] is not None and d['start_date'] is not None:
                d['active_length'] = d['end_date'] - d['start_date']
            else:
                d['active_length'] = None

            active_date_ranges.append(d)

        return active_date_ranges

    @classmethod
    def rank_sparsity(cls, master_df: pd.DataFrame, active_date_ranges: List[Tuple]) -> pd.DataFrame:
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
    def plot_skill_ts(cls, ts, skill_name):
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

    master_df = jackson_ggn_db.filter_active_work_sets(start_date='2017-01-01',
                                                       end_date='2019-03-23',
                                                       active_cutoff='2019-01-01',
                                                       percent_cutoff=90,
                                                       from_cache=True)
    master_df.iloc[:, 150].plot()
    plt.show()
    # jackson_ggn_db.plot_pareto(top_n=600)

    jackson_ggn_db.close()