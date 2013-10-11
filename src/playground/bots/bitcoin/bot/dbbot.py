from __future__ import division
from func import *
from dbadapter import *
from sys import exit
import random
#parameters
checkwalletconsistency=0
parttotrade=3 # buys or sells 1/parttotrade of the wallet amount
#refinecurrprice=0

class DbBot(object):
    def __init__(self, wallet, frequency, timewindow, freshprices):
        self.logstr = 'dbbot('+wallet+'):'
        print now(), self.logstr, wallet, frequency, timewindow
        self.wallet = wallet
        self.frequency = frequency
        self.timewindow = timewindow
        self.freshprices = freshprices

    def run_once(self):
        print now(), self.logstr, 'retrieving my wallet...'
        my_usd,my_btc=db_get_wallet(self.wallet)
        print now(), 'USD: ', my_usd, 'BTC: ',my_btc
        # now verifying that wallet is not outofsync with mtgox
        if (checkwalletconsistency==1):
            wallets = get_wallets()
            mtgox_usd = int(wallets['USD']['Balance']['value_int'])
            mtgox_btc = int(wallets['BTC']['Balance']['value_int'])
            if (my_usd>mtgox_usd) or (my_btc>mtgox_btc):
                print now(), 'Internal error, exiting bot: strategy wallet has more than mtgox wallet!' 'USD: ', my_usd, 'BTC: ',my_btc, 'mtgox_USD', mtgox_usd, 'mtgox_BTC', mtgox_btc
                exit()
            print now(), self.logstr, 'Wallet '+self.wallet+' is consistent with mtgox one.'
            print now(), self.logstr, 'Sleeping 180 seconds before attempting anything.'
            time.sleep(180+random.randrange(0,5));
        else:
            print now(),self.logstr, "wallet consistency check disabled."

        if (self.freshprices==1):
            bid = float(current_bid_price()/rusd)
            ask = float(current_ask_price()/rusd)
            curprice = (bid + ask) / 2
        else:
            curprice = db_get_last();
            bid = curprice
            ask = curprice

        print now(),self.logstr, "Current prices retrieved."
        # now retrieving all parameters to start trading decision
        thlow     = db_get_thlow(self.timewindow);
        thhigh    = db_get_thhigh(self.timewindow);
        avg       = db_get_avg(self.timewindow)
        vol       = db_get_vol()
        usdtobuy  = float(my_usd/parttotrade)
        btctosell = float(my_btc/parttotrade)

        print now(),self.logstr, "Current parameters retrieved."
        if (self.freshprices==1):
            db_store_ticker(curprice, thhigh, thlow, avg, 0, bid, ask, vol, 0)

        print now(),self.logstr, "Storing data into local database."

        print now(), '*** Parameters to start trading decision'
        print now(), 'Frequency: ', self.frequency, 'Timewindow: ', self.timewindow
        print now(), 'Curprice: ', curprice, '$ Th_low: ', thlow, '$ Th_high: ', thhigh, '$'
        print now(), 'Bid: ', bid, '$ Ask: ', ask, '$ Spread: ', (ask-bid), '$'
        print now(), 'USDtobuy: ', usdtobuy, '$ BTCtosell: ', btctosell, ' BTC'

        if (thlow>thhigh):
            print now(), 'Internal error, exiting bot: (thlow>thhigh) thlow: ', thlow, ' thhigh: ', thhigh
            exit()

        #print now(), self.logstr, 'Sleeping 180 seconds before taking trading decision.'
        #time.sleep(180+random.randrange(0,5));

        '''
        if refinecurrprice==1:
            print now(), self.logstr, 'Preliminary trading decision:'
            if (curprice<=thlow) and (usdtobuy>0.01):
              curprice = float(current_ask_price()/rusd);
              print now(), self.logstr, 'If buying, current ask price is ',curprice
            else:
              if (curprice>=thhigh) and (btctosell>0.001):
                 curprice = float(current_bid_price()/rusd);
                 print now(), self.logstr, 'If selling, current bid price is ',curprice
              else:
                 print now(), self.logstr, 'Doing no trade.'
        '''
        if (ask<=thlow) and (usdtobuy>0.01):
            print now(), '*** Decided to BUY at ', ask, '$'
            btctobuy = (usdtobuy/ask)
            print now(), ' Buying ', btctobuy, ' bitcoins...'
            resbuy = buy(btctobuy*rbtc)
            print 'Buy result: ', resbuy
            new_btc = my_btc + btctobuy
            new_usd = my_usd - float(btctobuy*ask)
            print now(), self.logstr, 'New wallet is approximately'
            print now(), 'USD: ', new_usd, 'BTC: ', new_btc
            db_store_wallet(self.wallet, new_btc, new_usd, 0)

        else:
            if (bid>=thhigh) and (btctosell>0.001):
                print now(), '*** Decided to SELL at ', bid, '$'
                print now(), ' Selling ', btctosell, ' bitcoins...'
                ressell = sell(btctosell*rbtc)
                print 'Sell result: ', ressell
                new_btc = my_btc - btctosell
                new_usd = my_usd + float(btctosell*bid)
                print now(), self.logstr, 'New wallet is approximately'
                print now(), 'USD: ', new_usd, 'BTC: ', new_btc
                db_store_wallet(self.wallet, new_btc, new_usd, 0)
            else:
                print now(), self.logstr, 'Decided to wait...'

    def run(self):
        while 1:
            try:
                self.run_once()
            except IOError as e:
                print "I/O error({0}): {1}".format(e.errno, e.strerror)
            except ValueError:
                print "Could not convert data to an integer."
            except:
                print "Unexpected error:", sys.exc_info()[0]
                #raise

            mysleep = (self.frequency*60) + random.randrange(0,300);
            print now(), self.logstr, 'Sleeping for '+str(mysleep)+' seconds...'
            time.sleep(mysleep)

