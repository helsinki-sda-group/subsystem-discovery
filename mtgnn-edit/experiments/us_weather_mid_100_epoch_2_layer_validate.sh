#!/bin/bash
module use /appl/local/csc/modulefiles/
module load pytorch
source venv/bin/activate
python -u train_multi_step.py --only_validate --epochs 100 --num_split 3 --subgraph_size 329 --layers 2 --learning_rate 1e-2 --batch_size 256 --num_nodes 987  --data ./data/us_weather.parquet --seq_in_len 256 --seq_out_len 64 --save ./save/usw-mid-100_epoch_2layer/