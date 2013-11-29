# This Python file uses the following encoding: iso-8859-15
import urllib2, json, datetime, time
from mtgox import mtgox
from conf import key, secret, proxy, rbtc, rusd
from dbadapter import db_store_ticker, db_store_trade, db_get_avg, db_get_thhigh, db_get_thlow, db_get_last, db_store_wallet, db_get_wallet, db_adjust_wallet_usd, db_adjust_wallet_btc

if proxy:
    myproxy = urllib2.ProxyHandler({'http': proxy})
    opener = urllib2.build_opener(myproxy)
    opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.4 (KHTML, like Gecko) Chrome/22.0.1229.26 Safari/537.4')]
    urllib2.install_opener(opener)

gox = mtgox(key, secret, 'funny-bot-bitcoin')

import traceback, cStringIO
def get_err():
    f = cStringIO.StringIO( )
    #traceback.print_exc(file=f)
    return f.getvalue( )

def now():
    return datetime.datetime.utcnow()

def ticker():
    return json.loads(urllib2.urlopen('http://data.mtgox.com/api/1/BTCUSD/ticker').read())['return']

def ticker2():
    res = json.loads(urllib2.urlopen('http://data.mtgox.com/api/2/BTCUSD/money/ticker').read())['data']
    db_store_ticker(res['last']['display_short'],
                    res['high']['display_short'],
                    res['low']['display_short'],
                    res['avg']['display_short'],
                    res['vwap']['display_short'],
                    res['buy']['display_short'],
                    res['sell']['display_short'],
                    res['vol']['display_short'])
    return res


def sync_wallets():
    info = gox.req('money/info', {})
    res = info['data']['Wallets']

    usd = res['USD']['Balance']['display'].replace("$","")
    #eur = res['EUR']['Balance']['display'].replace("","") # does not work, see first line with encoding
    eur = 0
    btc = res['BTC']['Balance']['display'].replace("BTC","")
    db_store_wallet('mtgox', btc, usd, eur)

    # compute adjustment
    total_usd,total_btc,total_bucket = db_get_wallet("total")
    mt_usd, mt_btc, mt_bucket = db_get_wallet("mtgox")

    adjust_usd = mt_usd - total_usd
    adjust_btc = mt_btc - total_btc

    print ""
    print "USD adjustment:", adjust_usd, "$"
    print "BTC adjustmnet:", adjust_btc, "BTC"

    if (adjust_usd==0) & (adjust_btc==0):
        print "mtgox and total wallet already synchronized :-)"
        return res


    print ""

    if db_adjust_wallet_usd("shortterm", adjust_usd)==1:
        print "USD adjustment done on shortterm portfolio"
    elif db_adjust_wallet_usd("midterm", adjust_usd)==1:
        print "USD adjustment done on midterm portfolio"
    elif db_adjust_wallet_usd("longterm", adjust_usd)==1:
        print "USD adjustment done on longterm portfolio"
    elif db_adjust_wallet_usd("tiz", adjust_usd)==1:
        print "USD adjustment done on tiz portfolio"
    else:
        print "Could not perform USD adjustment, try to move all USD in one portfolio!"
        adjust_usd=0

    if db_adjust_wallet_btc("shortterm", adjust_btc)==1:
        print "BTC adjustment done on shortterm portfolio"
    elif db_adjust_wallet_btc("midterm", adjust_btc)==1:
        print "BTC adjustment done on midterm portfolio"
    elif db_adjust_wallet_btc("longterm", adjust_btc)==1:
        print "BTC adjustment done on longterm portfolio"
    elif db_adjust_wallet_btc("tiz", adjust_btc)==1:
        print "BTC adjustment done on tiz portfolio"
    else:
        print "Could not perform BTC adjustment, try to move all BTC in one portfolio!"
        adjust_btc=0

    db_store_wallet('total', total_btc+adjust_btc, total_usd+adjust_usd, 0)

    return res

def get_orders():
    return gox.req('money/orders')['data']

def buy(amount, price=None):
    if price is None:
        res=gox.req('BTCUSD/money/order/add', {'amount_int': amount, 'type': 'bid'})
        myamount = float(amount)/rbtc
        return res
    else:
        res=gox.req('BTCUSD/money/order/add', {'amount_int': amount, 'type': 'bid', 'price_int': price})
        myamount = float(amount)/rbtc
        myprice  = float(price)/rusd
        return res

def sell(amount, price=None):
    if price is None:
        res=gox.req('BTCUSD/money/order/add', {'amount_int': amount, 'type': 'ask'})
        myamount = float(amount)/rbtc
        return res
    else:
        res=gox.req('BTCUSD/money/order/add', {'amount_int':amount, 'type': 'ask', 'price_int': price})
        myamount = float(amount)/rbtc
        myprice  = float(price)/rusd
        return res

def cancel(order_id):
    return gox.req('money/order/cancel', {'oid': order_id})

def cancel_all():
    res  = []
    for order in get_orders():
        res.append(cancel(order['oid']))
    return res

def get_order_result(ctype, order_id):
    #only for complete order
    #ctype: bid or ask
    return gox.req('money/order/result', {'type': ctype, 'order': order_id})

def lag():
    return gox.req('money/order/lag')

def quote(ctype, amount):
    return gox.req('BTCUSD/money/order/quote', {'amount': amount, 'type': ctype})

# error in interface bid/ask is inverted!
def current_bid_price():
    return quote('ask', rbtc)['data']['amount']

# error in interface bid/ask is inverted!
def current_ask_price():
    return quote('bid', rbtc)['data']['amount']

def get_avg():
    return db_get_avg()

def get_thlow():
    return db_get_thlow()

def get_thhigh():
    return db_get_thhigh()

def get_last():
    return db_get_last()

