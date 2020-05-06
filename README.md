# Context-Aware Model on Fairseq

The implementation of "Does Multi-Encoder Help? A Case Study on Context-Aware Neural Machine Translation"

> This code is based on [Fairseq v0.6.2](https://github.com/pytorch/fairseq/tree/v0.6.2)

## Installation

1. `pip install -r requirements.txt`
2. `python setup.py develop`
3. `python setup.py install`

## Prepare Training Data

> `bash runs/prepare-en2ru.sh`

## Train

### Train transformer baseline

> `bash runs/train-en2ru.sh baseline`

### Train context-aware model

> `bash runs/train-en2ru.sh inside-context`
>
> `bash runs/train-en2ru.sh outside-context`

### Train model with gaussian noise

> `bash runs/train-en2ru.sh gaussian`

## Infer

> `bash runs/translate-en2ru.sh baseline`
>
> `bash runs/translate-en2ru.sh inside-context`
>
> `bash runs/translate-en2ru.sh outside-context`
>
> `bash runs/translate-en2ru.sh gaussian`
