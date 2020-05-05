#!/usr/bin/env bash
set -e

data_URL="http://data.statmt.org/acl18_contextnmt_data"

mkdir -p data
mkdir -p data/token

echo 'download ...'
for i in dev test train
do

wget -P data/token $data_URL/${i}.en.context.gz
gzip data/token/${i}.en.context.gz -d
mv data/token/${i}.en.context data/token/${i}.context

wget -P data/token $data_URL/${i}.en.gz
gzip data/token/${i}.en.gz -d

wget -P data/token $data_URL/${i}.ru.gz
gzip data/token/${i}.ru.gz -d
done


num_operations=32000

mkdir -p data/bpe

cat data/token/train.en > data/token/all.en
cat data/token/train.context >> data/token/all.en

echo 'learn bpe...'
subword-nmt learn-bpe -s $num_operations < data/token/all.en > data/bpe/en.bpecode &
subword-nmt learn-bpe -s $num_operations < data/token/train.ru > data/bpe/ru.bpecode &
wait

echo 'apply bpe...'
for i in train dev test
do
    echo $i
    subword-nmt apply-bpe -c data/bpe/en.bpecode < data/token/${i}.context > data/bpe/${i}.context &
    subword-nmt apply-bpe -c data/bpe/en.bpecode < data/token/${i}.en > data/bpe/${i}.en &
    subword-nmt apply-bpe -c data/bpe/ru.bpecode < data/token/${i}.ru > data/bpe/${i}.ru &
    wait
done

echo 'bpe done'

python3 preprocess.py --source-lang en --target-lang ru --trainpref data/bpe/train --validpref data/bpe/dev --testpref data/bpe/test --destdir data-bin/sent --workers 32

python3 preprocess.py --source-lang en --target-lang ru --trainpref data/bpe/train --validpref data/bpe/dev --testpref data/bpe/test --destdir data-bin/context --workers 32 --srcdict data-bin/sent/dict.en.txt --tgtdict data-bin/sent/dict.ru.txt --context


