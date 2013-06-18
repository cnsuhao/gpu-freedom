import mysql.connector
from datetime import date, datetime, timedelta
from mtgox_key import mysql_host, mysql_username, mysql_password, mysql_database
create_user="bitcoinbot"

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
    
    '''
    query = ("select avg(price) from pricevalue where create_dt >= (NOW() - INTERVAL 9 DAY);")
    cursor.execute(query)
    myavg = cursor.fetchone()
    print myavg
    '''
    add_ticker = ("INSERT INTO pricevalue "
                  "(create_dt, price, high, low, volume, avgexchange, myavg, th_low, th_high, changepct, create_user) "
                  "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)")

    data_ticker = (mynow, last, high, low, vol, avg, 0, 0, 0, 0, create_user)
    
    cursor.execute(add_ticker, data_ticker)
    cnx.commit()
    
    cursor.close()
    cnx.close()
    print "ticker inserted into database"
