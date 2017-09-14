#!/bin/bash
# main page is ?
cd "$(dirname "$0")"

wget -O store_polorex_log --no-proxy http://127.0.0.1/gpu_freedom/src/playground/bots/gridcoin/polorexbot/polorex_grc.php 

cat store_polorex_log >> polorex.log
cat store_polorex_log


