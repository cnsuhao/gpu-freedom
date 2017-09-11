#!/bin/bash
# main page is ?
cd "$(dirname "$0")"

wget -O store_polorexcex_eth_log --no-proxy http://127.0.0.1/gpu_freedom/src/playground/bots/gridcoin/polorexcexbot/polorexcex_eth.php 

cat store_polorexcex_eth_log >> polorexcex_eth.log
cat store_polorexcex_eth_log

#php polorexcex_eth.php 
