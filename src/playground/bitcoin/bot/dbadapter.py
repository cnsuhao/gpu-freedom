import mysql.connector

from mtgox_key import mysql_host, mysql_username, mysql_password, mysql_database

def db_store_ticker(last, high, low, avg, vwap, buy, sell, vol):
    cnx = mysql.connector.connect(user=mysql_username, password=mysql_password,
                              host=mysql_host,
                              database=mysql_database)
    print "hello database"
    cnx.close()
