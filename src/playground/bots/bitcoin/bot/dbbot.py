from func import *
from dbadapter import *
from sys import exit
import random
#parameters
checkwalletconsistency=0
refinecurrprice=0
parttotrade=3 # buys or sells 1/parttotrade of the wallet amount

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
            print now(), self.logstr, 'Sleeping 125 seconds before attempting anything.'
            time.sleep(120+random.randrange(0,5));
        else:
            print now(),self.logstr, "wallet consistency check disabled."

        print now(),self.logstr, "retrieving mtgox ticker..."
        if (self.freshprices==1):
           ticker2()

        # now retrieving all parameters to start trading decision
        curprice  = db_get_last();
        thlow     = db_get_thlow(self.timewindow);
        thhigh    = db_get_thhigh(self.timewindow);
        usdtobuy  = float(my_usd/parttotrade)
        btctosell = float(my_btc/parttotrade)

        print now(), '*** Parameters to start trading decision'
        print now(), 'Frequency: ', self.frequency, 'Timewindow: ', self.timewindow
        print now(), 'Curprice: ', curprice, '$ Th_low: ', thlow, '$ Th_high: ', thhigh, '$'
        print now(), 'USDtobuy: ', usdtobuy, '$ BTCtosell: ', btctosell, ' BTC'

        if (thlow>thhigh):
            print now(), 'Internal error, exiting bot: (thlow>thhigh) thlow: ', thlow, ' thhigh: ', thhigh
            exit()

        print now(), self.logstr, 'Sleeping 125 seconds before taking trading decision.'
        time.sleep(120+random.randrange(0,5));

        if refinecurrprice==1:
            print now(), self.logstr, 'Preliminary trading decision:'
            if (curprice<=thlow) and (usdtobuy>0.01):
              curprice = float(current_ask_price())/rusd;
              print now(), self.logstr, 'If buying, current ask price is ',curprice
            else:
              if (curprice>=thhigh) and (btctosell>0.001):
                 curprice = float(current_bid_price())/rusd;
                 print now(), self.logstr, 'If selling, current bid price is ',curprice
              else:
                 print now(), self.logstr, 'Doing no trade.'

        if (curprice<=thlow) and (usdtobuy>0.01):
            print now(), '*** Decided to BUY'
            btctobuy = (usdtobuy/curprice)
            print now(), ' Buying ', btctobuy, ' bitcoins...'
            resbuy = buy(btctobuy*rbtc)
            print 'Buy result: ', resbuy
            new_btc = my_btc + btctobuy
            new_usd = my_usd - float(btctobuy*curprice)
            print now(), self.logstr, 'New wallet is approximately'
            print now(), 'USD: ', new_usd, 'BTC: ', new_btc
            db_store_wallet(self.wallet, new_btc, new_usd, 0)

        else:
            if (curprice>=thhigh) and (btctosell>0.001):
                print now(), '*** Decided to SELL'
                print now(), ' Selling ', btctosell, ' bitcoins...'
                ressell = sell(btctosell*rbtc)
                print 'Sell result: ', ressell
                new_btc = my_btc - btctosell
                new_usd = my_usd + float(btctosell*curprice)
                print now(), self.logstr, 'New wallet is approximately'
                print now(), 'USD: ', new_usd, 'BTC: ', new_btc
                db_store_wallet(self.wallet, new_btc, new_usd, 0)

        '''
        if self.next_action=='sell':
            current_price = current_ask_price()
            print now(), 'run_once', my_btc, my_usd, current_price, self.next_action, self.next_price
            amount = min(self.max_btc, my_btc)
            if current_price>=self.next_price or random.random()<=0.01:
                print now(), 'begin sell ', amount
                print 'sell result', sell(amount)
                self.next_action = 'dbbot:buy'
                self.next_price = int(current_price*(1+self.trigger_percent))
                print now(), 'sell ', amount
        elif self.next_action=='buy':
            current_price = current_bid_price()
            print now(), 'run_once', my_btc, my_usd, current_price, self.next_action, self.next_price
            money = min(self.max_usd, my_usd)
            amount = int(money*1.0/current_price*rbtc)
            if current_price<=self.next_price or random.random()>=0.99:
                print now(), 'begin buy', amount
                print 'buy result', buy(amount)
                self.next_action = 'dbbot:sell'
                self.next_price = int(current_price*(1+self.trigger_percent))
                print now(), 'buy', amount
        '''
    def run(self):
        while 1:
            try:
                self.run_once()
            except:
                print now(), self.logstr, "Error - ", get_err()

            mysleep = (self.frequency*60) + random.randrange(0,120);
            print now(), self.logstr, 'Sleeping for '+str(mysleep)+' seconds...'
            time.sleep(mysleep)

