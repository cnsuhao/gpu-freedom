#!/bin/bash
cd "$(dirname "$0")"
rm polorex.log.14
mv polorex.log.13 polorex.log.14
mv polorex.log.12 polorex.log.13
mv polorex.log.11 polorex.log.12
mv polorex.log.10 polorex.log.11
mv polorex.log.9 polorex.log.10
mv polorex.log.8 polorex.log.9
mv polorex.log.7 polorex.log.8
mv polorex.log.6 polorex.log.7
mv polorex.log.5 polorex.log.6
mv polorex.log.4 polorex.log.5
mv polorex.log.3 polorex.log.4
mv polorex.log.2 polorex.log.3
mv polorex.log.1 polorex.log.2
mv polorex.log polorex.log.1
touch polorex.log
chmod 666 *any.log*
