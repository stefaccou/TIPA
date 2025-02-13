#!/bin/bash
#SBATCH --cluster=wice
#SBATCH --partition=interactive
#SBATCH --account=lp_hpcinfo
#SBATCH --nodes=1 --ntasks=1
#SBATCH --time=2:00
#SBATCH --job-name=test_for_adapters
#SBATCH --output=%x.out

TASK=$1
DATA_SETUP=$2
TRAIN_LANG=$3
BASE_MODEL=$4
NONE_TR=$5
SEED=$6
EPOCHS=$7

mkdir ${VSC_DATA}/train_res
mkdir ${VSC_DATA}/eval_res

TRAIN_OUTPUT_DIR="${VSC_DATA}/train_res/${DATA_SETUP}_${TASK}_${TRAIN_LANG}_${BASE_MODEL}_NONE_TR=${NONE_TR}_SEED=${SEED}_EPOCHS=${EPOCHS}_DATE=$(date +"%Y-%m-%d_%H:%M")"
EVAL_OUTPUT_DIR="${VSC_DATA}/eval_res/${TASK}_${DATA_SETUP}_${TRAIN_LANG}_${BASE_MODEL}_NONE_TR=${NONE_TR}_SEED=${SEED}_EPOCHS=${EPOCHS}_DATE=$(date +"%Y-%m-%d_%H:%M")"
echo "TRAIN_OUTPUT_DIR: $TRAIN_OUTPUT_DIR"
echo "EVAL_OUTPUT_DIR: $EVAL_OUTPUT_DIR"

RUN_NAME="RUN_${TASK}_${DATA_SETUP}_${TRAIN_LANG}_${BASE_MODEL}_NONE_TR=${NONE_TR}_SEED=${SEED}_EPOCHS=${EPOCHS}_DATE=$(date +"%Y-%m-%d_%H:%M")"
echo "RUN_NAME: $RUN_NAME"

apptainer exec adapters-env.sif python run_adapters.py \
    --task $TASK \
    --data_setup $DATA_SETUP \
    --train_lang $TRAIN_LANG \
    --base_model $BASE_MODEL \
    --seed $SEED \
    --epochs $EPOCHS \
    --train_output_dir $TRAIN_OUTPUT_DIR \
    --eval_output_dir $EVAL_OUTPUT_DIR
