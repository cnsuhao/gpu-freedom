import MySQLdb

from mtgox_key import mysql_host, mysql_username, mysql_password, mysql_database

def db_store_ticker(last, high, low, avg, vwap, buy, sell, vol):
    db=MySQLdb.connect(mysql_host, mysql_username, mysql_password, mysql_database)
    print "hello database"
    db.close()