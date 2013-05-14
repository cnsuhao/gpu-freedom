#!/bin/bash
rm -rf bitcoincharts.html
wget -O bitcoincharts.html --no-check-certificate -U "Mozilla/5.0 (Windows NT 6.1; rv:2.0b7pre) Gecko/20100921 Firefox/4.0b7pre" http://www.bitcoincharts.com 
./removeif0.sh bitcoincharts.html
if [ -f bitcoincharts.html ] 
then
    wget -O store_bitcoin_value_bitcoincharts.html --no-proxy http://127.0.0.1:8090/bitcoin/store_bitcoin_value_bitcoincharts.php
fi