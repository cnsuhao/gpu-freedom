#!/bin/bash
# main page is ?
cd "$(dirname "$0")"
rm -rf data.csv
wget -O data.csv --no-check-certificate -U "Mozilla/5.0 (Windows NT 6.1; rv:2.0b7pre) Gecko/20100921 Firefox/4.0b7pre" "https://data.geo.admin.ch/ch.meteoschweiz.swissmetnet/VQHA69.csv"
./removeif0.sh data.csv
if [ -f data.csv ] 
then
    wget -O store_meteo_bafu --no-proxy http://127.0.0.1/gpu_freedom/src/playground/bots/meteo/store_meteo_bafu.php 
    cp ./data.csv ../../../../../
fi
