#!/bin/bash
rm -rf bitcoincharts.html
wget -O bitcoincharts.html --no-check-certificate -U "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.4 (KHTML, like Gecko) Chrome/22.0.1229.26 Safari/537.4" http://www.bitcoincharts.com 
./removeif0.sh bitcoincharts.html
if [ -f bitcoincharts.html ] 
then
    wget -O store_bitcoin_value_bitcoincharts.html --no-proxy http://127.0.0.1:8090/gpu_freedom/src/playground/bitcoin/store_bitcoin_value_bitcoincharts.php
fi