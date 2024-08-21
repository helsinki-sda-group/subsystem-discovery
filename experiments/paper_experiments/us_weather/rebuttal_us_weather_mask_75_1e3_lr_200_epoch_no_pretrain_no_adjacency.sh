#!/bin/bash

PYTHON_BIN="/pyenv/shims/python -u"
CODE_HOME=/users/sarapist/code/nasa-anomaly/
MODULE_NAME=models.jax.patchtst_equinox
SCRIPT_NAME=$SBATCH_SCRIPT_NAME

PARAMS="
                --seed 1337
                --batch_size 512
                --learning_rate 1e-3
                --epochs 0
                --masked_forecast_epochs 0
                --forecast_epochs 200
                --disable_adjacency

                --dataset_name us_weather

                --input_dim 256
                --pred_len 64

                --n_heads 8
                --n_layers 8
                --n_decoder_layers 1
                --mask_p 0.75
                --l1_regularize 0
                --n_forecast_layers 0

                --revin
                --instance_norm
                --rank_div 16

                --script_name $SCRIPT_NAME
"


# Remove newlines and ignore lines starting with #
PARAMS=$(echo "$PARAMS" | grep -v '^\s*//' | tr -d '\n')
#PARAMS="--seed 1337"

FULL_COMMAND="cd $CODE_HOME; $PYTHON_BIN -m $MODULE_NAME $PARAMS"
temp_file=$(mktemp)
echo "$FULL_COMMAND" > $temp_file

echo cmd: $FULL_COMMAND
echo tmp: $temp_file

singularity exec -B /project/project_462000406 -B /scratch/project_462000406 /scratch/project_462000406/anomaly_0.0.10.sif sh $temp_file

rm $temp_file
