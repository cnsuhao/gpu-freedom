#!/bin/bash
rm -rf bitcoin.html
wget -O bitcoin.html --no-check-certificate -U "Mozilla/5.0 (Windows NT 6.1; rv:2.0b7pre) Gecko/20100921 Firefox/4.0b7pre" http://www.mtgox.com 
./removeif0.sh bitcoin.html
if [ -f bitcoin.html ] 
then
    wget -O store_bitcoin_value_mtgox.html --no-proxy http://127.0.0.1:8090/bitcoin/store_bitcoin_value_mtgox.php
fi