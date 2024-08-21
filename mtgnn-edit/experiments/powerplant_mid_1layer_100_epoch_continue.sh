#!/bin/bash
module use /appl/local/csc/modulefiles/
module load pytorch
source venv/bin/activate
python -u train_multi_step.py --continue_train --epochs 100 --layers 1 --num_split 6 --subgraph_size 331 --learning_rate 1e-2 --batch_size 256 --num_nodes 1988 --data ./data/powerplant.parquet --seq_in_len 256 --seq_out_len 64 --save ./save/pp-mid-1layer-100epoch/
