#!/bin/bash
# main page is ?
rm -rf data.xml
wget -O data.xml --no-check-certificate -U "Mozilla/5.0 (Windows NT 6.1; rv:2.0b7pre) Gecko/20100921 Firefox/4.0b7pre" "http://www.hydrodaten.admin.ch/lhg/SMS.xml"
./removeif0.sh data.xml
if [ -f data.csv ] 
then
    wget -O store_meteo_bafu --no-proxy http://127.0.0.1/gpu_freedom/src/playground/bots/runofriver/store_runofriver.php
fi