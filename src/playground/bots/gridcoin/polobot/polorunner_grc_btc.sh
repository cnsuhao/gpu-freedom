#!/bin/bash
# main page is ?
cd "$(dirname "$0")"

wget -O store_polobot_log --no-proxy http://127.0.0.1/gpu_freedom/src/playground/bots/gridcoin/polobot_grc_btc.php 

cat store_polobot_log >> polobot.log
rm -rf store_polobot.log


