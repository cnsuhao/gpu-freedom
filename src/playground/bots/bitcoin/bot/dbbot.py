from func import *
import random

class DbBot(object):
    def __init__(self, wallet, frequency, timewindow):
        self.logstr = 'dbbot('+wallet+'):'
        print now(), self.logstr, wallet, frequency, timewindow
        self.wallet = wallet
        self.frequency = frequency
        self.timewindow = timewindow

    def run_once(self):
	    '''
        wallets = get_wallets()
        my_usd = int(wallets['USD']['Balance']['value_int'])
        my_btc = int(wallets['BTC']['Balance']['value_int'])
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

