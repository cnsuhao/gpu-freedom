#!/bin/bash
cd "$(dirname "$0")"

rm -rf arstechnica.html
wget -O arstechnica.html --no-check-certificate -U "Mozilla/5.0 (Windows NT 6.1; rv:2.0b7pre) Gecko/20100921 Firefox/4.0b7pre" "http://www.arstechnica.com"
./removeif0.sh arstechnica.html
if [ -f arstechnica.html ] 
then
    wget -O store_news_arstechnica.html --no-proxy http://127.0.0.1/gpu_freedom/src/playground/bots/news/store_news_arstechnica.php
fi