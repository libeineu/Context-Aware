#!/usr/bin/bash
set -e

model_root_dir=checkpoints

# set tag
tag=$1
flag=$2

# set device
gpu=0

# set dataset
who=test
#who=valid

if [ $tag == "baseline" ] || [ $tag == "gaussian" ]; then
        task=translation
        data_dir=sent
        batch_size=64
        beam=4
        length_penalty=0
elif [ $tag == "inside-context" ] || [ $tag == "outside-context" ]; then
        task=translation_context
        data_dir=context
        ensemble=
        batch_size=64
        beam=4
        length_penalty=0
else
        echo "unknown tag=$tag"
        exit
fi

model_dir=$model_root_dir/$tag

checkpoint=checkpoint_best.pt

cmd="python3 -u generate.py
data-bin/$data_dir
--task $task
--path $model_dir/$checkpoint
--gen-subset $who
--batch-size $batch_size
--beam $beam
--lenpen $length_penalty
--output $model_dir/translation.$data_dir.$who.unsort
--quiet
--remove-bpe"

if [ -n $flag ] && [ $flag == "ignore" ] && [ $task == "translation_context" ]; then
        cmd=${cmd}" --model-overrides {\'ignore\':True} "
fi

output=$model_dir/translation.$data_dir.$who.log

export CUDA_VISIBLE_DEVICES=$gpu
eval $cmd | tee $output

python3 rerank.py $model_dir/translation.$data_dir.$who.unsort $model_dir/translation.$data_dir.$who

