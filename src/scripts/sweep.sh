#!/bin/bash

Help() {
  echo "Syntax: sweep.sh [f|w|u|l|o|s|n|d|h]"
  echo ""
  echo "Options:"
  echo "f     huggingface token."
  echo "w     weights and biases token."
  echo "u     tpu host username."
  echo "l     scaling lock: one of aspect, depth, width."
  echo "o     optimizer name: one of adamw, lion, muon."
  echo "s     dataset name: one of fineweb, commonpile, cci3hq."
  echo "n     param count N."
  echo "d     token budget D."
  echo "h     Print this help."
  echo
}

while getopts "f:w:u:l:o:s:n:d:h" option; do
  case $option in
    f)
      HF_TOKEN=$OPTARG;;
    w)
      WB_TOKEN=$OPTARG;;
    u)
      TPU_HOST_USERNAME=$OPTARG;;
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


for BSZ in 32768 65536 131072 262144 524288 1048576 2097152;
do
    for LR in 0.0078125 0.005524271728019903 0.00390625 0.0027621358640099515 0.001953125 0.0013810679320049757 0.0009765625 0.0006905339660024879 0.00048828125 0.00034526698300124393 0.000244140625 0.00017263349150062197 0.0001220703125;
    do
        echo "Starting run with bsz $BSZ and lr $LR";
        /home/$TPU_HOST_USERNAME/.local/bin/uv run src/babel/main.py \
            --group="$SCALING_LOCK-$OPTIM_NAME-$DATASET_NAME-$N-$D" \
            --config="src/babel/configs/base.py" \
            --workdir="gs://tpu_persist_bucket/babel_sweeps/" \
            --hf_token="$HF_TOKEN" \
            --wb_token="$WB_TOKEN" \
            --config.scaling_lock="$SCALING_LOCK" \
            --config.optim_name="$OPTIM_NAME" \
            --config.dataset_name="$DATASET_NAME" \
            --config.model_size="$N" \
            --config.token_budget="$D" \
            --config.tokens_per_global_batch="$BSZ" \
            --config.lr_eta="$LR";
    done;
done;