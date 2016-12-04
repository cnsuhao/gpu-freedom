#!/bin/bash
cd "$(dirname "$0")"

sudo rm -f dump_bots_14.tar.gz
sudo mv dump_bots_13.tar.gz dump_bots_14.tar.gz
sudo mv dump_bots_12.tar.gz dump_bots_13.tar.gz
sudo mv dump_bots_11.tar.gz dump_bots_12.tar.gz
sudo mv dump_bots_10.tar.gz dump_bots_11.tar.gz
sudo mv dump_bots_09.tar.gz dump_bots_10.tar.gz
sudo mv dump_bots_08.tar.gz dump_bots_09.tar.gz
sudo mv dump_bots_07.tar.gz dump_bots_08.tar.gz
sudo mv dump_bots_06.tar.gz dump_bots_07.tar.gz
sudo mv dump_bots_05.tar.gz dump_bots_06.tar.gz
sudo mv dump_bots_04.tar.gz dump_bots_05.tar.gz
sudo mv dump_bots_03.tar.gz dump_bots_04.tar.gz
sudo mv dump_bots_02.tar.gz dump_bots_03.tar.gz
sudo mv dump_bots_01.tar.gz dump_bots_02.tar.gz
sudo mv dump_bots.tar.gz dump_bots_01.tar.gz
