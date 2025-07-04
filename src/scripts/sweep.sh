#!/bin/bash

Help() {
  echo "Syntax: sweep.sh [w|f|b|u|t|q|l|o|s|n|d|h]"
  echo ""
  echo "Options:"
  echo "w     workdir."
  echo "f     huggingface token."
  echo "b     weights and biases token."
  echo "u     uv path."
  echo "t     using a tpu cluster?"
  echo "l     scaling lock: one of aspect, depth, width."
  echo "o     optimizer name: one of adamw, lion, muon."
  echo "s     dataset name: one of fineweb, commonpile, cci3hq."
  echo "n     param count N."
  echo "d     token budget D."
  echo "h     Print this help."
  echo
}

while getopts "w:f:b:u:t:l:o:s:n:d:h" option; do
  case $option in
    w)
      WORKDIR=$OPTARG;;
    f)
      HF_TOKEN=$OPTARG;;
    b)
      WB_TOKEN=$OPTARG;;
    u)
      UV_PATH=$OPTARG;;
    t)
      TPU_CLUSTER=$OPTARG;;
    l)
      SCALING_LOCK=$OPTARG;;
    o)
      OPTIM_NAME=$OPTARG;;
    s)
      DATASET_NAME=$OPTARG;;
    n)
      N=$OPTARG;;
    d)
      D=$OPTARG;;
    h)
      Help
      exit;;
    \?)
      echo "Parse error. Run -h for help."
      exit;;
  esac
done


for BSZ in 32768 65536 131072 262144;
do
  for LR in 0.00390625 0.001953125 0.0009765625 0.00048828125 0.00024414062;
  do
    for WD in 0.00390625 0.001953125 0.0009765625 0.00048828125 0.00024414062;
    do
      echo "Starting run with bsz $BSZ, lr $LR, wd $WD";
      "$UV_PATH" run src/babel/main.py \
        --group="$SCALING_LOCK-$OPTIM_NAME-$DATASET_NAME-$N-$D" \
        --config="src/babel/configs/base.py" \
        --workdir="$WORKDIR" \
        --hf_token="$HF_TOKEN" \
        --wb_token="$WB_TOKEN" \
        --tpu="$TPU_CLUSTER" \
        --config.scaling_lock="$SCALING_LOCK" \
        --config.optim_name="$OPTIM_NAME" \
        --config.dataset_name="$DATASET_NAME" \
        --config.model_size="$N" \
        --config.token_budget="$D" \
        --config.tokens_per_global_batch="$BSZ" \
        --config.lr_eta="$LR" \
        --config.wd_lam="$WD" \
        --config.wd_indep=True \
        --config.qk_norm=True; 
    done;
  done;
done;
