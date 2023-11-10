#!/bin/bash

python training/exp_runner.py --conf confs_sg/tensoir.conf --data_split_dir "../data/tensoir/$1" --expname "tensoir_$1" --trainstage IDR --gpu $2
python training/exp_runner.py --conf confs_sg/tensoir.conf --data_split_dir "../data/tensoir/$1" --expname "tensoir_$1" --trainstage Illum --gpu $2
python training/exp_runner.py --conf confs_sg/tensoir.conf --data_split_dir "../data/tensoir/$1" --expname "tensoir_$1" --trainstage Material --gpu $2

