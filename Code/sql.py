import mysql.connector
from mysql.connector import errorcode
import settings as s

def connect(db=s.DB_NAME):
    cnx = mysql.connector.connect(host='localhost',\
                                database=db,\
                                user='root',\
                                password="airbnb2018",\
                                auth_plugin='mysql_native_password',\
                                charset='utf8'\
                                )
    return cnx

def create_database(db=s.DB_NAME):
    cnx = connect('mysql')
    cursor = cnx.cursor()
    try:
        cursor.execute(
            "CREATE DATABASE {} DEFAULT CHARACTER SET 'utf8'".format(db))
    except mysql.connector.Error as err:
        print("Failed creating database: {}".format(err))
        exit(1)
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
