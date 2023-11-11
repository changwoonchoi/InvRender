#!/bin/bash

python scripts/relight.py --conf confs_sg/tensoir.conf --data_split_dir "../data/tensoir/$1" --expname "tensoir_$1" --timestamp latest --gpu $2

