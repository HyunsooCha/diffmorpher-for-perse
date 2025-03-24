#!/bin/zsh
set -e

source $HOME/conda/etc/profile.d/conda.sh
conda activate

############################
DEVICE=0
ENV_NAME=perse_avatar
WORK_DIR=$HOME/GitHub/perse-dev
DATASET_NAME=shrink_man
DATASET_A=0000_beard_balbo_unkempt_black_soft_portrait
DATASET_B=0001_beard_balbo_short_white_curly_portrait
IMAGE_A=0001
IMAGE_B=0001
OUTPUT_PATH=./results/$DATASET_A-$IMAGE_A-to-$DATASET_B-$IMAGE_B
LORA_CKPT=5
############################

conda activate $ENV_NAME

MODEL_NAME=stabilityai/stable-diffusion-2-1-base
DATASET_PATH=$WORK_DIR/data/datasets/$DATASET_NAME/synthetic_dataset
IMAGE_A_PATH=$DATASET_PATH/$DATASET_A/image/$IMAGE_A.jpg
IMAGE_B_PATH=$DATASET_PATH/$DATASET_B/image/$IMAGE_B.jpg
LORA_PATH_A=$DATASET_PATH/$DATASET_A/lora_$LORA_CKPT.ckpt
LORA_PATH_B=$DATASET_PATH/$DATASET_B/lora_$LORA_CKPT.ckpt

cd $WORK_DIR'/code/submodules/diffmorpher-for-perse'
export CUDA_VISIBLE_DEVICES=$DEVICE

python main_vis.py --model_path $MODEL_NAME \
                   --image_path_0 $IMAGE_A_PATH \
                   --image_path_1 $IMAGE_B_PATH \
                   --output_path $OUTPUT_PATH \
                   --load_lora_path_0 $LORA_PATH_A \
                   --load_lora_path_1 $LORA_PATH_B \
                   --use_adain \
                   --save_inter