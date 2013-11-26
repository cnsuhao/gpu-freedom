import sys

from func import *
from stablebot import StableBot
from rampbot import RampBot
from dbbot import DbBot
from conf import version, th_day_interval
from dbadapter import *

if __name__=='__main__':
    try:
        if len(sys.argv)==1:
            print "For help, type python ./main.py help"

        elif sys.argv[1]=='wallets':
            wallets = get_wallets()
            for wallet in wallets:
                print wallet, wallets[wallet]['Balance']['display']

        elif sys.argv[1]=='orders':
            for order in get_orders():
                print order['oid'], order['status'], order['type'], 'price: ', order['price']['display'], \
                    'amount: ', order['amount']['display']

        elif sys.argv[1]=='buy':
            usdtobuy=float(sys.argv[2])
            my_wallet=sys.argv[3]
            if len(sys.argv)>=5:
                price = int(float(sys.argv[4])*rusd)
                ask=float(sys.argv[4])
            else:
                price = None
                ask=db_get_ask()
            my_usd,my_btc,my_bucket_usd=db_get_wallet(my_wallet)
            if usdtobuy>my_usd:
                print 'Error, I have only', my_usd, '$ available...'
            else:
                btctobuy=float(usdtobuy/ask)
                if btctobuy<0.01:
                    print 'Error, need to buy at least 0.01 BTC, you tried to buy '+btctobuy
                else:
                    amount = int(float(btctobuy)*rbtc)
                    print buy(amount, price)
                    new_btc = my_btc + btctobuy
                    new_usd = my_usd - float(btctobuy*ask)
                    print now(), 'New wallet is approximately'
                    print now(), 'USD: ', new_usd, 'BTC: ', new_btc
                    db_store_wallet(my_wallet, new_btc, new_usd, 0, my_bucket_usd)
                    db_store_trade('BUY', btctobuy, ask, 1, my_wallet)

        elif sys.argv[1]=='sell':
            btctosell=float(sys.argv[2])
            my_wallet=sys.argv[3]
            amount = int(btctosell*rbtc)
            my_usd,my_btc,my_bucket_usd=db_get_wallet(my_wallet)
            if len(sys.argv)>=5:
                price = int(float(sys.argv[4])*rusd)
                bid = sys.argv[4]
            else:
                price = None
                bid = db_get_bid()

            if btctosell<0.01:
                print 'Error, need to sell at least 0.01 BTC!'
            elif my_btc<btctosell:
                print 'Error, I have only', my_btc, 'BTC available...'
            else:
                print sell(amount, price)
                new_btc = my_btc - btctosell
                new_usd = my_usd + float(btctosell*bid)
                print now(), 'New wallet is approximately'
                print now(), 'USD: ', new_usd, 'BTC: ', new_btc
                db_store_wallet(my_wallet, new_btc, new_usd, 0, my_bucket_usd)
                db_store_trade('SELL', btctosell, bid, 1, my_wallet)

        elif sys.argv[1]=='move_btc':
            btctomove=float(sys.argv[2])
            from_wallet=sys.argv[3]
            to_wallet=sys.argv[4]

            from_usd,from_btc,from_bucket_usd=db_get_wallet(from_wallet)
            if btctomove>from_btc:
                print 'Error, can not move so many BTC, max is ', from_btc
            else:
                to_usd,to_btc,to_bucket_usd=db_get_wallet(to_wallet)
                to_btc = to_btc + btctomove
                from_btc = from_btc - btctomove
                db_store_wallet(from_wallet, from_btc, from_usd, 0, from_bucket_usd)
                db_store_wallet(to_wallet, to_btc, to_usd, 0, to_bucket_usd)


        elif sys.argv[1]=='cancel':
            order_id = sys.argv[2]
            print cancel(order_id)

        elif sys.argv[1]=='cancel_all':
            print cancel_all()

        elif sys.argv[1]=='result':
            ctype = sys.argv[2]    #bid or ask
            order_id = sys.argv[3]
            print get_order_result(ctype, order_id)

        elif sys.argv[1]=='ticker':
            res = ticker2()
            for k in ['last', 'high', 'low', 'avg', 'vwap', 'buy', 'sell', 'vol']:
                print k, res[k]['display_short']

        elif sys.argv[1]=='lag':
            print lag()

        elif sys.argv[1]=='quote':
            ctype = sys.argv[2]    #bid or ask
            amount = int(float(sys.argv[3])*rbtc)
            print quote(ctype, amount)

        elif sys.argv[1]=='stablebot':
            max_btc = int(float(sys.argv[2])*rbtc)
            max_usd = int(float(sys.argv[3])*rusd)
            init_action = sys.argv[4] #sell, buy
            init_price = int(float(sys.argv[5])*rusd)
            trigger_percent = float(sys.argv[6])
            bot = StableBot(max_btc, max_usd, init_action, init_price, trigger_percent)
            bot.run()
        elif sys.argv[1]=='rampbot':
            max_btc = int(float(sys.argv[2])*rbtc)
            max_usd = int(float(sys.argv[3])*rusd)
            init_action = sys.argv[4] #sell, buy
            init_price = int(float(sys.argv[5])*rusd)
            trigger_percent = float(sys.argv[6])
            rampbot = RampBot(max_btc, max_usd, init_action, init_price, trigger_percent)
            rampbot.run()
        elif sys.argv[1]=='dbbot':
            mywallet = sys.argv[2]
            myfrequency = int(sys.argv[3])
            mytimewindow = int(sys.argv[4])
            freshprice = int(sys.argv[5])
            dbbot = DbBot(mywallet, myfrequency, mytimewindow, freshprice)
            dbbot.run()
        elif sys.argv[1]=='thresholds':
            print " days:  "+str(th_day_interval)
            print " high:  "+str(get_thhigh())
            print " avg:   "+str(get_avg())
            print " low:   "+str(get_thlow())
            print " last:  "+str(get_last())

        elif sys.argv[1]=='help':
            print "***********************"
            print "* tiz bitcoin bot "+version+' *'
            print "***********************"
            print "Adapted from perol's funny bitcoinbot available "
            print "at http://github.com/perol/funny-bot-bitcoin"
            print ""
            print "Usage:"
            print " python main.py ticker"
            print " python main.py wallets"
            print " python main.py orders"
            print " python main.py thresholds"
            print " python main.py buy 40 [wallet] {price}"
            print " python main.py sell 0.01 [wallet] {price}"
            print " python main.py move_btc 0.01 [from_wallet] [to_wallet]"
            print " python main.py move_usd 50 [from_wallet] [to_wallet]"
            print " python main.py stablebot 0.01 2 buy 110.0 0.01"
            print " python main.py rampbot 0.01 2 buy 110.0 0.01"
            print " python main.py dbbot [wallet] [frequency in minutes] [time window in minutes] [freshprices]"
            print " python main.py dbbot shortterm 2 180 [3 hours] 1"
            print " python main.py dbbot midterm 5 360 [6 hours] 0"
            print " python main.py dbbot longterm 10 1440 [1 days] 0"
            print ""
            print "Warning: this bot is Jack of all trades and master of none!"
            print "         Use at your own risk :-)"

        else:
            pass

    except Exception as e:
        print now(), "Error - ", get_err()



