import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Jackson_GGN_DB(object):

    def __init__(self):
        try:
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



if __name__ == '__main__':
    jackson_ggn_db = Jackson_GGN_DB()
    cursor = jackson_ggn_db.cursor
    connection = jackson_ggn_db.connection
    query = "select \
                    start_ts, work_set_id, demand_context_id, count(*) as row_count, \
                    sum(item_count_nb) as item_count_nb_sum, sum(amount_nb) as amount_nb_sum \
                    from demand where demand_type_c = \'CURRENT\' \
                    and work_set_id = 125723 and demand_context_id in \
                    (select distinct on (start_ts) demand_context_id \
                    from demand where demand_type_c = \'CURRENT\' and work_set_id = 125723) \
                    group by work_set_id, start_ts, demand_context_id;"
    df = pd.read_sql(query, connection)
    sns.lineplot(df['start_ts'], df['item_count_nb_sum'])
    plt.xlabel('Time')
    plt.ylabel('item_count_nb_sum')
    plt.show()
    cursor.close()
    jackson_ggn_db.connection.close()
