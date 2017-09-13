#!/bin/bash
# main page is ?
cd "$(dirname "$0")"

wget -O store_polorexcex_any_log --no-proxy http://127.0.0.1/gpu_freedom/src/playground/bots/gridcoin/polorexcexbot/polorexcex_any.php 

cat store_polorexcex_any_log >> polorexcex_any.log
cat store_polorexcex_any_log

#php polorexcex_any.php 
