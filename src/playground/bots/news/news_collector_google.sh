#!/bin/bash
cd "$(dirname "$0")"

rm -rf news.html
wget -O news.html --no-check-certificate -U "Mozilla/5.0 (Windows NT 6.1; rv:2.0b7pre) Gecko/20100921 Firefox/4.0b7pre" "https://news.google.com/news"
./removeif0.sh news.html
if [ -f news.html ] 
then
    wget -O store_news_google.html --no-proxy http://127.0.0.1/gpu_freedom/src/playground/bots/news/store_news_google.php
fi