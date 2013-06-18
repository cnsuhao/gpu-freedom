import mysql.connector
from datetime import date, datetime, timedelta
from conf import mysql_host, mysql_username, mysql_password, mysql_database, th_day_interval
create_user="bitcoinbot"


def db_get_avg():
    cnx = mysql.connector.connect(user=mysql_username, password=mysql_password,
                              host=mysql_host,
                              database=mysql_database)
    cursor = cnx.cursor()
    query = ("select avg(price) from pricevalue where create_dt >= (NOW() - INTERVAL "+th_day_interval+" DAY);")
    cursor.execute(query)
    myavg = cursor.fetchone()
    cursor.close()
    cnx.close()
    return myavg[0]
    
def db_get_thlow():
    cnx = mysql.connector.connect(user=mysql_username, password=mysql_password,
                              host=mysql_host,
                              database=mysql_database)
    cursor = cnx.cursor()
    query = ("select min(price) from pricevalue where create_dt >= (NOW() - INTERVAL "+th_day_interval+" DAY);")
    cursor.execute(query)
    thlow = cursor.fetchone()
    cursor.close()
    cnx.close()
    return thlow[0]

def db_get_thhigh():
    cnx = mysql.connector.connect(user=mysql_username, password=mysql_password,
                              host=mysql_host,
                              database=mysql_database)
    cursor = cnx.cursor()
    query = ("select max(price) from pricevalue where create_dt >= (NOW() - INTERVAL "+th_day_interval+" DAY);")
    cursor.execute(query)
    thhigh = cursor.fetchone()
    cursor.close()
    cnx.close()
    return thhigh[0]
    

def db_store_ticker(last, high, low, avg, vwap, buy, sell, vol):
    cnx = mysql.connector.connect(user=mysql_username, password=mysql_password,
                              host=mysql_host,
                              database=mysql_database)
    cursor = cnx.cursor()

    last  = last.replace("$","")
    high = high.replace("$","")
    low  = low.replace("$","")
    avg = avg.replace("$","")
    vwap = vwap.replace("$","")
    buy = buy.replace("$","")
    sell = sell.replace("$","")
    vol = vol.replace("BTC","")
    vol = vol.replace(",","")
    mynow = datetime.now().date()
    myavg = db_get_avg()
    thlow = db_get_thlow()
    thhigh = db_get_thhigh()
    
    add_ticker = ("INSERT INTO pricevalue "
                  "(create_dt, price, high, low, volume, avgexchange, myavg, th_low, th_high, changepct, create_user) "
                  "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)")

    data_ticker = (mynow, last, high, low, vol, avg, myavg, thlow, thhigh, 0, create_user)
    
    cursor.execute(add_ticker, data_ticker)
    cnx.commit()
    
    cursor.close()
    cnx.close()
   #print "ticker inserted into database"
