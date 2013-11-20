#!/bin/bash
# main page is http://www.swissgrid.ch/content/swissgrid/de/home/experts/topics/frequency.html 
rm -rf frequency.html
wget -O frequency.html --no-check-certificate -U "Mozilla/5.0 (Windows NT 6.1; rv:2.0b7pre) Gecko/20100921 Firefox/4.0b7pre" "http://swissgrid.ch/mvc.do/applicationFrequencyTimeAjax?tooltipText=Live-Daten (Aktualisierung alle 30 Sekunden, 20 Minuten zeitversetzt)&deviationText=Aktuelle Netzabweichung&frequencyText=Aktuelle Frequenz&_charset_=UTF-8"
./removeif0.sh frequency.html
if [ -f frequency.html ] 
then
    wget -O store_frequency_swissgrid.html --no-proxy http://127.0.0.1/gpu_freedom/src/playground/bots/powergrid/store_frequency_swissgrid.php
fi