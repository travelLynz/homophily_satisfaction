import mysql.connector
from sqlalchemy import create_engine
from mysql.connector import errorcode
import settings as s
import pandas as pd

def connect(db=s.DB_NAME):
    cnx = mysql.connector.connect(host='localhost',\
                                database=db,\
                                user='root',\
                                password="airbnb2018",\
                                auth_plugin='mysql_native_password',\
                                charset='utf8mb4'\
                                )
    return cnx

def create_database(db=s.DB_NAME):
    cnx = connect('mysql')
    cursor = cnx.cursor()
    try:
        cursor.execute(
            "CREATE DATABASE {} DEFAULT CHARACTER SET 'utf8mb4'".format(db))
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_DB_CREATE_EXISTS:
            print("Database already exists.")
        else:
            print(err.msg)
    finally:
        cursor.close()
        cnx.close()

def use_database(cnx, db=s.DB_NAME):
    try:
        cnx.database = db
        return cnx
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_BAD_DB_ERROR:
            create_database(db)
            cnx.database = db
            return cnx
        else:
            print(err)
            exit(1)

def create_tables(db=s.DB_NAME):
    cnx = connect(db)
    cursor = cnx.cursor()
    for name, ddl in s.TABLES.items():
        create_table(name, ddl, cursor)
    cursor.close()
    cnx.close()

def create_table(name, ddl, cursor):
    try:
        print("Creating table {}: ".format(name), end='')
        cursor.execute(ddl)
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
            print("already exists. Recreating table")
            cursor.execute("DROP TABLE " + name + ';')
            create_table(name, ddl, cursor)
        else:
            print(err.msg)
    else:
        print("OK")
def create_alchemy_engine(con=s.CONNECT_SQLALCHEMY_STRING,pool_size=100, max_overflow=20, pool_timeout=30):
    return create_engine(con, pool_size=pool_size, max_overflow=max_overflow, pool_timeout=pool_timeout)

def push_tables_to_db(tables, table_names):
    engine = create_alchemy_engine()
    for t, n in zip(tables, table_names):
        print('Pushing \'%s\' records' % n)
        t.to_sql(name=n, con=engine, if_exists='append',index=False, chunksize=100)

def get_neighborhood(con, neighborhood):
    return pd.read_sql('SELECT * FROM listings where neighbourhood_group_cleansed = "' + neighborhood + '";', con, chunksize=None)

def get_hosts(con, host_ids):
    return pd.read_sql('SELECT DISTINCT * FROM hosts where id in (' + str(host_ids)[1:-1] + ');', con, chunksize=None)

def get_hosts_trips(con, host_ids):
    return pd.read_sql('SELECT DISTINCT * FROM hostTrips where hostId in (' + str(host_ids)[1:-1] + ');', con, chunksize=None)

def get_guest_reviews(con, listing_ids):
    return pd.read_sql('SELECT DISTINCT * FROM guestReviews where listing_id in (' + str(listing_ids)[1:-1] + ');', con, chunksize=None)

def get_host_reviews(con):
    return pd.read_sql_table('hostReviews', con, chunksize=None)

def get_guests(con):
    return pd.read_sql_table('guests', con, chunksize=None)

def get_manhattan_data():
    engine = create_alchemy_engine()

    #Get Listings
    listings = get_neighborhood(engine, 'Manhattan')
    print('Retrieved %d Manhattan listings' % len(listings))

    #Get Reviews
    listing_ids = set(listings['id'])
    reviews = get_guest_reviews(engine, listing_ids)
    print('Retrieved %d Manhattan reviews' % len(reviews))

    #Get Hosts
    host_ids = set(listings['host_id'])
    hosts = get_hosts(engine, host_ids)
    hosts = hosts[~hosts['id'].isnull()]
    print('Retrieved %d Manhattan hosts' % len(hosts))
    hosts_with_reviews = reviews['recipient_id'].unique()
    print('Retrieved only %d (%.2f%%) Manhattan hosts with reviews' % (len(hosts_with_reviews), len(hosts_with_reviews)*100/len(hosts)))

    # Host hostTrips
    host_trips = get_hosts_trips(engine, host_ids)
    print('Retrieved %d Manhattan host trips' % len(host_trips))

    #Get Host Reviews
    guest_ids = set(reviews['reviewer_id'])
    host_reviews = get_host_reviews(engine)
    print('Retrieved %d reviews from hosts who have hosted Manhattan Guests' % len(host_reviews))
    print('Total Number of %d guests that have reviewed Manhattan listings' % len(guest_ids))

    guests = get_guests(engine)
    man_guests = guests[guests['id'].isin(guest_ids)]
    man_guests = man_guests.drop_duplicates(subset="id", keep="last").reset_index(drop=True)
    print('Retrieved %d (%.2f%%) guests that have reviewed Manhattan listings' % (len(man_guests), len(man_guests)*100/len(guest_ids)))

    return (listings, reviews, hosts, man_guests, host_trips, host_reviews)
