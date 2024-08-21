#!/bin/bash
module use /appl/local/csc/modulefiles/
module load pytorch
source venv/bin/activate
python train_multi_step.py --epochs 1000 --layers 2 --learning_rate 1e-3 --num_nodes 12 --subgraph_size 12 --data ./data/pendulum.txt --seq_in_len 256 --seq_out_len 64 --save ./save/pendulum-1k/
