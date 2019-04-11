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

    def get_ws_time_series_by_skill(self, skill_name, start_time, end_time, from_cache=False):
        if from_cache:
            df = pd.read_csv(self.cache_path + '/ws_time_series_%s.csv' % skill_name)
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
            df.to_csv(self.cache_path + '/ws_time_series_%s.csv' % skill_name)
            return df

    def get_summed_ws_time_series_by_skill(self, start_time, end_time, from_cache=False):
        if from_cache:
            df = pd.read_csv(self.cache_path + '/summed_ws_time_series_by_skill.csv')
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
            df.to_csv(self.cache_path + '/summed_ws_time_series_by_skill.csv')
            return df

    def get_summed_work_items_by_work_set(self, start_time, end_time, from_cache=False):
        if from_cache:
            df = pd.read_csv(self.cache_path + '/summed_work_items_by_work_set.csv')
            return df
        else:
            query = "SELECT tmp.work_set_id, count(tmp.work_item_id) as work_item_count, \
            to_timestamp(floor(extract(EPOCH FROM tmp.received_ts) / 1800) * 1800) AS rec_ts_rounded \
            FROM (SELECT work_item_id, work_set_id, received_ts, ROW_NUMBER() OVER (PARTITION BY work_item_id ORDER BY received_ts ASC) rn \
            FROM work_item_log WHERE received_ts >= '%s' and received_ts <= '%s') tmp\
            WHERE rn = 1 group by work_set_id, rec_ts_rounded;" % (start_time, end_time)
            df = pd.read_sql(query, self.connection)
            df.to_csv(self.cache_path + '/summed_work_items_by_work_set.csv')
            return df


    def get_summed_work_items_by_skill(self, start_time, end_time, from_cache=False):
        pass


if __name__ == '__main__':
    jackson_ggn_db = Jackson_GGN_DB(cache_path='./data')
    # df = jackson_ggn_db.get_ws_time_series_by_skill(skill_name='POS Annuity Disbursement File Review Manual',
    #                                                 start_time='2018-12-31',
    #                                                 end_time='2019-01-31',
    #                                                 from_cache=False)
    # df.pivot(index='start_ts', columns='work_set_id', values='item_count_nb_sum').sum(axis=1).plot()
    df = jackson_ggn_db.get_summed_work_items_by_work_set(start_time='2018-12-31',
                                                           end_time='2019-01-31',
                                                           from_cache=False)
    df.pivot(index='rec_ts_rounded', columns='work_set_id', values='work_item_count').plot(legend=None)
    plt.show()
    jackson_ggn_db.close()
