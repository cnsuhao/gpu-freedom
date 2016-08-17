#!/bin/bash
cd "$(dirname "$0")"

rm -rf cnn.html
wget -O cnn.html --no-check-certificate -U "Mozilla/5.0 (Windows NT 6.1; rv:2.0b7pre) Gecko/20100921 Firefox/4.0b7pre" "http://www.cnn.com"
./removeif0.sh cnn.html
if [ -f cnn.html ] 
then
    wget -O store_news_cnn.html --no-proxy http://127.0.0.1/gpu_freedom/src/playground/bots/news/store_news_cnn.php
fi