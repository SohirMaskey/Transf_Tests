#!/bin/bash

eval "$(conda shell.bash hook)"
cd src
conda activate pyg_cuda102
python classifier.py --train --batch 8 --graph_size 1000 --data_size 50 --K 2 --model "ChebNet" --epochs 300

