#!/bin/bash
# main page is ?
cd "$(dirname "$0")"

wget -O store_polorexcex_log --no-proxy http://127.0.0.1/gpu_freedom/src/playground/bots/gridcoin/polorexcexbot/polorexcex_grc.php 

cat store_polorexcex_log >> polorexcex.log
cat store_polorexcex_log


