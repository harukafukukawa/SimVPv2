#!/bin/bash
# This script is used to try different parameters for simvp
batch_sizes=(4 8 16)
b_index=$((RANDOM % ${#batch_sizes[@]}))
b=${batch_sizes[$b_index]}

drops=(0.0 0.0001 0.001 0.01 0.1 0.5 1.0 2.0)
d_index=$((RANDOM % ${#drops[@]}))
d=${drops[$d_index]}

epochs=(250 300 350)
e_index=$((RANDOM % ${#epochs[@]}))
e=${epochs[$e_index]}

optimizers=("sgd" "adam" "momentum" "adamw" "nadam" "radam" "adamp" "rmsprop" "rmsproptf" "nvnovograd")
o_index=$((RANDOM % ${#optimizers[@]}))
o=${optimizers[$o_index]}

weight_decays=(0.0 0.00001 0.0001 0.001 0.01)
w_index=$((RANDOM % ${#weight_decays[@]}))
w=${weight_decays[$w_index]}

lrs=(0.0001 0.0005 0.001 0.005 0.01 0.02)
l_index=$((RANDOM % ${#lrs[@]}))
l=${lrs[$l_index]}

lr_k_decays=(0.0005 0.001 0.01 0.1 0.5 1.0 2.0)
k_index=$((RANDOM % ${#lr_k_decays[@]}))
k=${lr_k_decays[$k_index]}

warmup_lr=(0.0001 0.001 0.01 0.1 0.5 1.0)
wlr_index=$((RANDOM % ${#warmup_lr[@]}))
wlr=${warmup_lr[$wlr_index]}

min_lr=(0.00000001 0.0000001 0.000001 0.00001 0.0001)
minlr_index=$((RANDOM % ${#min_lr[@]}))
minlr=${min_lr[$minlr_index]}

configs=("E3DLSTM.py" "SAConvLSTM.py" "E3DLSTM.py" "SAConvLSTM.py" "E3DLSTM.py" "SAConvLSTM.py" "E3DLSTM.py" "SAConvLSTM.py" "E3DLSTM.py" "SAConvLSTM.py" "E3DLSTM.py" "SAConvLSTM.py" "E3DLSTM.py" "SAConvLSTM.py" "E3DLSTM.py" "SAConvLSTM.py" "E3DLSTM.py" "SAConvLSTM.py" "ConvLSTM.py" "E3DLSTM.py" "PredRNN.py" "PredRNNpp.py" "PredRNNv2.py" "SimVP.py")
#configs=("E3DLSTM.py" "SAConvLSTM.py")
configs_index=$((RANDOM % ${#configs[@]}))
config=${configs[$configs_index]}

echo python tools/non_dist_train.py -d cracks --lr $l --lr_k_decay $k -b $b --drop $d --epoch $e --opt $o --weight_decay $w --warmup_lr $wlr --min_lr $minlr -c ./configs/cracks/$config --ex_name cracks_jun_30_${config:0:-3}-${l}-${k}-${b}-${d}-${e}-${o}-${w}-${wlr}-${minlr}; 
python tools/non_dist_train.py -d cracks --lr $l --lr_k_decay $k -b $b --drop $d --epoch $e --opt $o --weight_decay $w --warmup_lr $wlr --min_lr $minlr -c ./configs/cracks/$config --ex_name cracks_jun_30_${config:0:-3}-${l}-${k}-${b}-${d}-${e}-${o}-${w}-${wlr}-${minlr}; 
