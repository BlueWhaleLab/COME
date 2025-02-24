#!/bin/bash
# ImageNet1K Path (if EATA method is included)
data='/path/to/dataset/Imagenet1K'
# ImageNet_C Path (necessary)
data_corruption='/path/to/dataset/ImageNet_C'
# Log File Output Path 
output='/path/to/output/result'
# open-world data path
ood_root='/data1/kongxinke/datasets/'
# corrupt level of ImageNet_C
level=3
exp_type="open-world"
step=1
# backbone model
model="vitbase_timm"
seed=2024
ood_rate=0.5
test_batch_size=64
export CUDA_VISIBLE_DEVICES=1

run_experiment () {
  local method=$1
  local scoring_function=$2
  local name="experiment_${method}_${model}_ood${ood_rate}_level${level}_seed${seed}_${exp_type}"
  echo "Running $name with seed: $seed"
  python3 main.py --data $data --data_corruption $data_corruption --output $output --ood_root $ood_root\
    --method $method --level $level --exp_type $exp_type --step $step\
    --ood_rate $ood_rate --scoring_function $scoring_function --model $model --seed $seed --test_batch_size $test_batch_size
}
methods=("no_adapt" "Tent" "EATA" "SAR" "Tent_COME" "EATA_COME" "SAR_COME")
scoring_functions=( "msp"  "msp" "msp" "msp" "dirichlet" "dirichlet" "dirichlet" )

for i in ${!methods[@]}; do
  run_experiment "${methods[$i]}" "${scoring_functions[$i]}"
done
