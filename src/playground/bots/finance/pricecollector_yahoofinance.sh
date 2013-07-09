#!/bin/bash
rm -rf finance.html
wget -O finance.html --no-check-certificate -U "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.4 (KHTML, like Gecko) Chrome/22.0.1229.26 Safari/537.4" "http://finance.yahoo.com/q?s=^$1"
./removeif0.sh finance.html
if [ -f finance.html ] 
then
    wget -O store_yahoo_ticker.html --no-proxy "http://127.0.0.1:8090/gpu_freedom/src/playground/bots/finance/store_yahoo_ticker.php?ticker=$1"
fi