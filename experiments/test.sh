#!/bin/bash

PYTHON_BIN="/Users/tesatesa/devaus/subsystem-discovery-high-dimensional-time-series-masked-autoencoders/venv/bin/python -u"
CODE_HOME="/Users/tesatesa/devaus/subsystem-discovery-high-dimensional-time-series-masked-autoencoders"
MODULE_NAME=run
#SCRIPT_NAME=$SBATCH_SCRIPT_NAME
SCRIPT_NAME="test_script"

PARAMS="
                --seed 1337
                --batch_size 128
                --learning_rate 1e-3
                --epochs 2
                --masked_forecast_epochs 0
                --forecast_epochs 0

                --dataset_name us_weather

                --input_dim 256
                --pred_len 64

                --n_heads 4
                --n_layers 1
                --n_decoder_layers 1
                --mask_p 0.75
                --l1_regularize 0
                --n_forecast_layers 0

                --revin
                --instance_norm
                --rank_div 1

                --script_name $SCRIPT_NAME
"

# Remove newlines and ignore lines starting with #
PARAMS=$(echo "$PARAMS" | grep -v '^\s*//' | tr -d '\n')

FULL_COMMAND="cd $CODE_HOME; $PYTHON_BIN -m $MODULE_NAME $PARAMS"

temp_file=$(mktemp)
echo "$FULL_COMMAND" > $temp_file

echo "Running command: $FULL_COMMAND"

sh $temp_file