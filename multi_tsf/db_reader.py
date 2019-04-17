import psycopg2
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class Jackson_GGN_DB(object):

    def __init__(self, cache_path) -> None:
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

    def get_all_skills(self):
        query = "select distinct(skill_display_nm) from work_skill_log;"
        df = pd.read_sql(query, self.connection)
        return df

    def get_all_work_sets(self):
        query = "select distinct(work_set_id) from work_set_log;"
        df = pd.read_sql(query, self.connection)
        return df

    def get_summed_demand_time_series_by_work_set_in_skill(self, skill_name, start_time, end_time, from_cache=False):
        if from_cache:
            df = pd.read_csv(self.cache_path + '/demand_time_series_%s.csv' % skill_name)
            return df
        else:
            query = "SELECT ts.time_stamp, js. *, skill_nm.skill_display_nm \
            FROM (SELECT * FROM   generate_series(timestamp '%s'::timestamp, timestamp '%s'::timestamp, interval  '30 min'::interval) time_stamp \
            WHERE extract(hour from time_stamp) >= 6 and extract(hour from time_stamp) <= 23) ts \
            LEFT JOIN (select start_ts, work_set_id, avg(item_count_nb_sum) as item_count_nb_sum, avg(estimated_completion_time_hrs) as estimated_completion_time_hrs from\
            (select start_ts, work_set_id, demand_context_id, sum(item_count_nb) as item_count_nb_sum, sum(amount_nb) / (60 * 60 * 1000) as estimated_completion_time_hrs \
            from demand where demand_type_c = 'CURRENT' and start_ts > '%s' and start_ts <= '%s' group by work_set_id, start_ts, demand_context_id) A group by A.work_set_id, A.start_ts) js\
            on ts.time_stamp = js.start_ts \
            LEFT JOIN\
            (select distinct on (skill_display_nm, work_set_id) skill_display_nm, work_set_id from work_skill_log a \
            JOIN work_set_log b on a.work_skill_id = b.work_skill_id) skill_nm ON js.work_set_id = skill_nm.work_set_id \
            where skill_nm.skill_display_nm = '%s';" % (start_time, end_time, start_time, end_time, skill_name)
            df = pd.read_sql(query, self.connection)
            df.to_csv(self.cache_path + '/demand_time_series_%s.csv' % skill_name)
            return df

    def get_summed_demand_time_series_by_skill(self, start_time, end_time, from_cache=False):
        if from_cache:
            df = pd.read_csv(self.cache_path + '/summed_demand_time_series_by_skill.csv')
            return df
        else:
            query = "SELECT start_ts, skill_display_nm, sum(item_count_nb_sum) as total_item_count, sum(estimated_completion_time_hrs) as total_est_completion_hrs \
                        FROM ((SELECT * FROM   generate_series(timestamp '%s'::timestamp, timestamp '%s'::timestamp, interval  '30 min'::interval) time_stamp \
                        WHERE extract(hour from time_stamp) >= 6 and extract(hour from time_stamp) <= 23) ts \
                        LEFT JOIN (select start_ts, work_set_id, avg(item_count_nb_sum) as item_count_nb_sum, avg(estimated_completion_time_hrs) as estimated_completion_time_hrs from\
                        (select start_ts, work_set_id, demand_context_id, sum(item_count_nb) as item_count_nb_sum, sum(amount_nb) / (60 * 60 * 1000) as estimated_completion_time_hrs \
                        from demand where demand_type_c = 'CURRENT' and start_ts > '%s' and start_ts <= '%s' group by work_set_id, start_ts, demand_context_id) A group by A.work_set_id, A.start_ts) js\
                        on ts.time_stamp = js.start_ts \
                        LEFT JOIN\
                        (select distinct on (skill_display_nm, work_set_id) skill_display_nm, work_set_id from work_skill_log a \
                        JOIN work_set_log b on a.work_skill_id = b.work_skill_id) skill_nm ON js.work_set_id = skill_nm.work_set_id) \
                        ws_time_series group by start_ts, skill_display_nm" % (start_time, end_time, start_time, end_time)
            df = pd.read_sql(query, self.connection)
            df.to_csv(self.cache_path + '/summed_demand_time_series_by_skill.csv')
            return df

    def get_summed_work_items_by_work_set(self, start_time, end_time, from_cache=False):
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
            on skill_nm.work_set_id = wil_cs.work_set_id; " % (start_time, end_time)
            df = pd.read_sql(query, self.connection)
            df.dropna(inplace=True)
            df = df.pivot(index='received_ts_rounded', columns='work_set_id', values='rec_item_count')
            df_timestamps = pd.DataFrame(pd.date_range(start_time, end_time, freq='30T'), columns=['received_ts_rounded'])
            df = df_timestamps.join(df, how='left', on='received_ts_rounded')
            del df_timestamps
            df = df.fillna(value=0)
            df.index = pd.to_datetime(df.index)
            df.to_csv(self.cache_path + '/summed_work_items_by_work_set.csv')
            return df


    def get_summed_work_items_by_skill(self, start_time, end_time, from_cache=False):
        if from_cache:
            df = pd.read_csv(self.cache_path + '/summed_work_items_by_skill.csv', index_col=0)
            df = df.set_index('received_ts_rounded')
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
            group by wil_cs.received_ts_rounded, skill_nm.skill_display_nm;" % (start_time, end_time)
            df = pd.read_sql(query, self.connection)
            df.dropna(inplace=True)
            df = df.pivot(index='received_ts_rounded', columns='skill_display_nm', values='rec_item_count')
            df_timestamps = pd.DataFrame(pd.date_range(start_time, end_time, freq='30T'), columns=['received_ts_rounded'])
            df = df_timestamps.join(df, how='left', on='received_ts_rounded')
            del df_timestamps
            df = df.fillna(value=0)
            df.index = pd.to_datetime(df.index)
            df.to_csv(self.cache_path + '/summed_work_items_by_skill.csv')
            return df


if __name__ == '__main__':
    jackson_ggn_db = Jackson_GGN_DB(cache_path='./data')
    skill_ts = jackson_ggn_db.get_summed_work_items_by_skill(start_time='2018-08-31',
                                                           end_time='2019-01-31',
                                                           from_cache=False)
    most_common_skills = skill_ts.sum(axis=0).sort_values(ascending=False)
    skill_ts = skill_ts[most_common_skills.index.tolist()[0:10]]
    skill_ts.plot()
    plt.show()
    jackson_ggn_db.close()
