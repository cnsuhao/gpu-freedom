from __future__ import division
from func import *
from dbadapter import *
from sys import exit
import random

class StopLossBot(object):
    def __init__(self, wallet, frequency, freshprices, stoploss):
        self.logstr = 'stoplossbot('+wallet+'):'
        print now(), self.logstr, wallet, frequency
        self.wallet = wallet
        self.frequency = frequency
        self.freshprices = freshprices
        self.stoploss = stoploss

    def run_once(self):
        print now(), self.logstr, 'retrieving my wallet...'
        my_usd,my_btc,my_bucket_usd=db_get_wallet(self.wallet)
        print now(), 'USD: ', my_usd, '$ BTC: ',my_btc, 'Bucket: ', my_bucket_usd, '$'

        if (self.freshprices==1):
            bid = float(current_bid_price()/rusd)
            ask = float(current_ask_price()/rusd)
        else:
            bid = db_get_bid();
            ask = db_get_ask();

        curprice = float((bid + ask) / 2)

        print now(),self.logstr, "Current prices retrieved."

        print 'Current price', curprice, ' USD'
        print 'BTCtosell: ', my_btc, ' BTC'
        print 'Stop loss:', self.stoploss, 'USD'
        print now(), '********************************************'
        print

        if (curprice<self.stoploss):
            print now(), 'Price is ', price, 'and is lower than stoploss! ', stoploss
            print now(), 'Selling immediately ', my_btc, 'BTC'
            ressell = sell(my_btc*rbtc)
            print 'Sell result: ', ressell
            db_store_wallet(self.wallet, 0, my_usd+float(my_btc*curprice), 0, my_bucket_usd)
            db_store_trade('SELL', my_btc, curprice, 1, self.wallet)
            db_store_total_wallet()
            os._exit()
        else:
             print "Nothing to do :-), sleep well!"



    def run(self):
        while 1:
            try:
                self.run_once()
            except IOError as e:
                print "I/O error({0}): {1}".format(e.errno, e.strerror)
            except ValueError:
                print "Could not convert data to an integer."
            except:
                print "Unexpected error!" #, sys.exc_info()[0]
                print "Error({0}): {1}".format(e.errno, e.strerror)
                #raise

            mysleep = (self.frequency*60) + random.randrange(0,120);
            print now(), self.logstr, 'Sleeping for '+str(mysleep)+' seconds...'
            time.sleep(mysleep)

