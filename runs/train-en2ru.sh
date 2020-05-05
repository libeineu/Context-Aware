#! /usr/bin/bash
set -e

device=0,1,2,3

tag=$1

if [ $tag == "baseline" ]; then
        task=translation
        arch=transformer_t2t_wmt_en_de
        share_embedding=0
        share_decoder_input_output_embed=1
        criterion=label_smoothed_cross_entropy
        fp16=1
        lr=0.0007
        warmup=4000
        dropout=0.3
        adam_betas="'(0.9, 0.997)'"
        max_tokens=4096
        update_freq=1
        weight_decay=0.0
        keep_last_epochs=1
        max_epoch=30
        max_update=
        data_dir=sent
        src_lang=en
        tgt_lang=ru
elif [ $tag == "inside-context" ]; then
        task=translation_context
        arch=in_context_transformer_t2t_wmt_en_de
        pretrained_model=checkpoints/baseline/checkpoint_best.pt
        context_layer=1
        share_embedding=0
        share_decoder_input_output_embed=1
        criterion=label_smoothed_cross_entropy
        fp16=1
        lr=0.0001
        warmup=4000
        dropout=0.3
        adam_betas="'(0.9, 0.997)'"
        max_tokens=4096
        update_freq=1
        weight_decay=0.0
        keep_last_epochs=1
        max_epoch=5
        max_update=
        data_dir=context
        src_lang=en
        tgt_lang=ru
elif [ $tag == "outside-context" ]; then
        task=translation_context
        arch=out_context_transformer_t2t_wmt_en_de
        pretrained_model=checkpoints/baseline/checkpoint_best.pt
        context_layer=1
        share_embedding=0
        share_decoder_input_output_embed=1
        criterion=label_smoothed_cross_entropy
        fp16=1
        lr=0.0001
        warmup=4000
        dropout=0.3
        adam_betas="'(0.9, 0.997)'"
        max_tokens=4096
        update_freq=1
        weight_decay=0.0
        keep_last_epochs=1
        max_epoch=5
        max_update=
        data_dir=context
        src_lang=en
        tgt_lang=ru
elif [ $tag == "gaussian" ]; then
        task=translation
        arch=rand_noise_transformer_t2t_wmt_en_de
        pretrained_model=checkpoints/baseline/checkpoint_best.pt
        share_embedding=0
        share_decoder_input_output_embed=1
        criterion=label_smoothed_cross_entropy
        fp16=1
        lr=0.0001
        warmup=4000
        dropout=0.3
        adam_betas="'(0.9, 0.997)'"
        max_tokens=4096
        update_freq=1
        weight_decay=0.0
        keep_last_epochs=1
        max_epoch=10
        max_update=
        data_dir=sent
        src_lang=en
        tgt_lang=ru
else
        echo "unknown tag=$tag"
        exit
fi

save_dir=checkpoints/$tag

if [ ! -d $save_dir ]; then
        mkdir -p $save_dir
fi
cp ${BASH_SOURCE[0]} $save_dir/train.sh

gpu_num=`echo "$device" | awk '{split($0,arr,",");print length(arr)}'`

cmd="python3 -u train.py data-bin/$data_dir
  --task $task
  --distributed-world-size $gpu_num -s $src_lang -t $tgt_lang
  --arch $arch
  --optimizer adam --clip-norm 0.0
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $warmup
  --lr $lr --min-lr 1e-09
  --weight-decay $weight_decay
  --criterion $criterion  --label-smoothing 0.1
  --max-tokens $max_tokens
  --update-freq $update_freq
  --no-progress-bar
  --log-interval 100
  --ddp-backend no_c10d 
  --save-dir $save_dir
  --keep-last-epochs $keep_last_epochs
  --tensorboard-logdir $save_dir"

cmd=${cmd}" --adam-betas "${adam_betas}
if [ $share_embedding -eq 1 ]; then
cmd=${cmd}" --share-all-embeddings "
fi
if [ $share_decoder_input_output_embed -eq 1 ]; then
cmd=${cmd}" --share-decoder-input-output-embed "
fi
if [ -n "$max_epoch" ]; then
cmd=${cmd}" --max-epoch "${max_epoch}
fi
if [ -n "$max_update" ]; then
cmd=${cmd}" --max-update "${max_update}
fi
if [ -n "$dropout" ]; then
cmd=${cmd}" --dropout "${dropout}
fi
if [ $fp16 -eq 1 ]; then
cmd=${cmd}" --fp16 "
fi
if [ -n "$pretrained_model" ]; then
cmd=${cmd}" --pretrained-path ${pretrained_model} "
fi
if [ -n "$context_layer" ]; then
cmd=${cmd}" --context-encoder-layers "${context_layer}
fi


export CUDA_VISIBLE_DEVICES=$device
cmd="nohup "${cmd}" >> $save_dir/train.log 2>&1 &"
eval $cmd
tail -f $save_dir/train.log
