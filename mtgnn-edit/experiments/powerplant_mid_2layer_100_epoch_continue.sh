#!/bin/bash
module use /appl/local/csc/modulefiles/
module load pytorch
source venv/bin/activate
python -u train_multi_step.py --epochs 100 --layers 2 --num_split 6 --subgraph_size 331 --learning_rate 1e-2 --batch_size 128 --num_nodes 1988 --data ./data/powerplant.parquet --seq_in_len 256 --seq_out_len 64 --save ./save/pp-mid-2layer-100epoch/
