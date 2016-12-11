#!/bin/bash
cd "$(dirname "$0")"

sudo rm -f dump_bots.sql
sudo rm -f dump_bots.tar 
sudo rm -f dump_bots.tar.gz

sudo mysqldump -u mybots -pbots13? --databases meteo news powergrid raspi runofriver > dump_bots.sql
sudo tar -cf dump_bots.tar dump_bots.sql
sudo gzip dump_bots.tar
scp dump_bots.tar.gz pi@firstlight:backup/
scp dump_bots.tar.gz dm@deepmind:backup/
